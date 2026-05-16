package tui

import (
	tea "github.com/charmbracelet/bubbletea"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

type progressScreen struct {
	deps Deps
	req  domain.Request
}

func newProgress(d Deps, req domain.Request) *progressScreen {
	return &progressScreen{deps: d, req: req}
}
func (p *progressScreen) Init() tea.Cmd                    { return nil }
func (p *progressScreen) Update(msg tea.Msg) (screen, tea.Cmd) { return p, nil }
func (p *progressScreen) View() string                     { return "progress stub — T4" }
