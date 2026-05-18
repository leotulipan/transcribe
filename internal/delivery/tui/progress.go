package tui

import (
	"context"
	"fmt"
	"strings"

	"github.com/charmbracelet/bubbles/progress"
	"github.com/charmbracelet/bubbles/spinner"
	tea "github.com/charmbracelet/bubbletea"

	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

type progressScreen struct {
	deps   Deps
	req    domain.Request
	ctx    context.Context
	cancel context.CancelFunc

	job      ports.Job
	bar      progress.Model
	spin     spinner.Model
	log      []string
	finished bool
	result   *domain.Result
	err      error
}

func newProgress(d Deps, req domain.Request) *progressScreen {
	ctx, cancel := context.WithCancel(context.Background())
	return &progressScreen{
		deps:   d,
		req:    req,
		ctx:    ctx,
		cancel: cancel,
		bar:    progress.New(progress.WithDefaultGradient()),
		spin:   spinner.New(spinner.WithSpinner(spinner.Dot)),
	}
}

func (p *progressScreen) Init() tea.Cmd {
	return tea.Batch(
		p.spin.Tick,
		func() tea.Msg {
			j, err := p.deps.Service.Submit(p.ctx, p.req)
			if err != nil {
				return doneMsg{err: err}
			}
			return startedMsg{job: j}
		},
	)
}

func (p *progressScreen) Update(msg tea.Msg) (screen, tea.Cmd) {
	switch m := msg.(type) {
	case spinner.TickMsg:
		var c tea.Cmd
		p.spin, c = p.spin.Update(m)
		return p, c
	case progress.FrameMsg:
		var c tea.Cmd
		updated, c := p.bar.Update(m)
		p.bar = updated.(progress.Model)
		return p, c
	case startedMsg:
		p.job = m.job
		return p, listenProgress(p.job)
	case progressMsg:
		p.log = append(p.log, fmt.Sprintf("[%s] %s", domain.Stage(m.Stage), m.Message))
		var cmds []tea.Cmd
		if m.Percent >= 0 {
			cmds = append(cmds, p.bar.SetPercent(m.Percent))
		}
		cmds = append(cmds, listenProgress(p.job))
		return p, tea.Batch(cmds...)
	case doneMsg:
		p.finished = true
		p.result = m.result
		p.err = m.err
		p.cancel()
		return p, func() tea.Msg { return finishedMsg{result: m.result, err: m.err} }
	case tea.KeyMsg:
		if m.Type == tea.KeyEsc || m.Type == tea.KeyCtrlC {
			if p.job != nil {
				p.job.Cancel()
			}
			p.cancel()
		}
	}
	return p, nil
}

func (p *progressScreen) View() string {
	var b strings.Builder
	b.WriteString(titleStyle.Render("Transcribing " + p.req.InputPath))
	b.WriteString("\n\n")
	if !p.finished {
		b.WriteString(p.spin.View() + " ")
		b.WriteString(p.bar.View())
		b.WriteString("\n\n")
	}
	// last 8 log lines
	start := 0
	if len(p.log) > 8 {
		start = len(p.log) - 8
	}
	for _, line := range p.log[start:] {
		b.WriteString(line + "\n")
	}
	if p.finished {
		if p.err != nil {
			b.WriteString("\n" + errorStyle.Render("error: "+p.err.Error()))
		} else if p.result != nil {
			preview := p.result.Text
			if len(preview) > 200 {
				preview = preview[:200] + "..."
			}
			b.WriteString("\n" + successStyle.Render("done"))
			if preview != "" {
				b.WriteString("\n" + preview)
			}
		}
	}
	return b.String()
}
