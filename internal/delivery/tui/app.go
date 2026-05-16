package tui

import (
	"context"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

// Prefill carries values already supplied via CLI flags so the TUI can skip
// asking for them.
type Prefill struct {
	InputPath string
	Provider  domain.ProviderID
	Model     string
	Language  string
	Formats   []domain.OutputFormat
}

type Deps struct {
	Service ports.TranscribeService
	Config  ports.Config
	Logger  ports.Logger
}

// App is the top-level Bubble Tea model. It delegates Update/View to whichever
// sub-screen is currently active.
type App struct {
	deps   Deps
	pre    Prefill
	cur    screen
	curID  screenID
	final  *domain.Result
	err    error
	width  int
	height int
}

func NewApp(deps Deps, pre Prefill) *App {
	a := &App{deps: deps, pre: pre}
	a.advanceFromInputs()
	return a
}

// advanceFromInputs picks the right starting screen given prefilled values.
func (a *App) advanceFromInputs() {
	switch {
	case a.pre.InputPath == "":
		a.curID = screenFilePicker
		a.cur = newFilePicker(a.pre)
	case a.pre.Provider == "" || len(a.pre.Formats) == 0:
		a.curID = screenOptions
		a.cur = newOptions(a.deps, a.pre)
	default:
		a.curID = screenProgress
		a.cur = newProgress(a.deps, a.buildRequest())
	}
}

func (a *App) buildRequest() domain.Request {
	return domain.Request{
		InputPath: a.pre.InputPath,
		Provider:  a.pre.Provider,
		Model:     a.pre.Model,
		Language:  a.pre.Language,
		Formats:   a.pre.Formats,
		UseCache:  true,
	}
}

func (a *App) Init() tea.Cmd { return a.cur.Init() }

func (a *App) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch m := msg.(type) {
	case tea.WindowSizeMsg:
		a.width, a.height = m.Width, m.Height
	case tea.KeyMsg:
		if m.Type == tea.KeyCtrlC || (m.Type == tea.KeyEsc && a.curID != screenProgress) {
			return a, tea.Quit
		}
	case advanceMsg:
		a.pre = m.pre
		a.advanceFromInputs()
		return a, a.cur.Init()
	case finishedMsg:
		a.final, a.err = m.result, m.err
		return a, tea.Quit
	}
	next, cmd := a.cur.Update(msg)
	a.cur = next
	return a, cmd
}

func (a *App) View() string { return a.cur.View() }

// Run starts the Bubble Tea program. Returns the final result (or nil) and
// any error from the pipeline.
func Run(ctx context.Context, deps Deps, pre Prefill) (*domain.Result, error) {
	a := NewApp(deps, pre)
	p := tea.NewProgram(a, tea.WithContext(ctx))
	if _, err := p.Run(); err != nil {
		return nil, err
	}
	return a.final, a.err
}

// advanceMsg propagates updated Prefill values when a sub-screen completes.
type advanceMsg struct{ pre Prefill }

// finishedMsg is emitted by the progress screen on pipeline completion.
type finishedMsg struct {
	result *domain.Result
	err    error
}
