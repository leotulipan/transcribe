package tui

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"

	assemblyaiclient "github.com/leotulipan/transcribe/internal/adapters/api/assemblyai"
	elevenlabsclient "github.com/leotulipan/transcribe/internal/adapters/api/elevenlabs"
	geminiclient "github.com/leotulipan/transcribe/internal/adapters/api/gemini"
	groqclient "github.com/leotulipan/transcribe/internal/adapters/api/groq"
	mistrakclient "github.com/leotulipan/transcribe/internal/adapters/api/mistral"
	openaiclient "github.com/leotulipan/transcribe/internal/adapters/api/openai"
	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

// wizardOrder is the canonical sequence of providers the wizard walks through.
var wizardOrder = []domain.ProviderID{
	domain.ProviderGroq,
	domain.ProviderOpenAI,
	domain.ProviderAssemblyAI,
	domain.ProviderElevenLabs,
	domain.ProviderGemini,
	domain.ProviderMistral,
}

// providerFactory constructs a ports.Provider from an ID and key.
// Replaceable in tests to avoid real network calls.
var providerFactory = newProviderForKey

func newProviderForKey(id domain.ProviderID, key string) (ports.Provider, error) {
	switch id {
	case domain.ProviderGroq:
		return groqclient.New(key, nil), nil
	case domain.ProviderOpenAI:
		return openaiclient.New(key, nil), nil
	case domain.ProviderAssemblyAI:
		return assemblyaiclient.New(key, nil), nil
	case domain.ProviderElevenLabs:
		return elevenlabsclient.New(key, nil), nil
	case domain.ProviderGemini:
		return geminiclient.New(key, nil), nil
	case domain.ProviderMistral:
		return mistrakclient.New(key, nil), nil
	default:
		return nil, fmt.Errorf("unknown provider: %s", id)
	}
}

type wizardState int

const (
	stateAskKey  wizardState = iota // showing text input for current provider
	stateTestKey                    // key test in flight
	stateSave                       // all providers done; confirm save
	stateDone                       // save confirmed; wizard exits
)

type keyResult struct {
	ok  bool
	msg string
}

// wizardScreen implements the screen interface for the interactive key setup wizard.
type wizardScreen struct {
	cfg       ports.Config
	ctx       context.Context
	providers []domain.ProviderID
	idx       int
	state     wizardState
	input     textinput.Model
	results   map[domain.ProviderID]keyResult
}

// checkDoneMsg is the async result of a CheckKey network call.
type checkDoneMsg struct {
	provider domain.ProviderID
	key      string
	ok       bool
	msg      string
}

// wizardDoneMsg signals the wizard has finished (saved or cancelled).
type wizardDoneMsg struct {
	cfg       ports.Config
	cancelled bool
}

func newWizardScreen(cfg ports.Config) *wizardScreen {
	if cfg.APIKeys == nil {
		cfg.APIKeys = map[domain.ProviderID]string{}
	}
	w := &wizardScreen{
		cfg:       cfg,
		ctx:       context.Background(),
		providers: wizardOrder,
		results:   map[domain.ProviderID]keyResult{},
	}
	w.input = newKeyInput(cfg.APIKeys[wizardOrder[0]])
	return w
}

func newKeyInput(current string) textinput.Model {
	ti := textinput.New()
	ti.Placeholder = "paste API key (Enter=test, Tab=skip)"
	ti.EchoMode = textinput.EchoPassword
	ti.EchoCharacter = '*'
	ti.SetValue(current)
	ti.Focus()
	return ti
}

func (w *wizardScreen) currentProvider() domain.ProviderID {
	return w.providers[w.idx]
}

func (w *wizardScreen) Init() tea.Cmd { return textinput.Blink }

func (w *wizardScreen) Update(msg tea.Msg) (screen, tea.Cmd) {
	switch w.state {
	case stateAskKey:
		return w.updateAskKey(msg)
	case stateTestKey:
		return w.updateTestKey(msg)
	case stateSave:
		return w.updateSave(msg)
	}
	return w, nil
}

func (w *wizardScreen) updateAskKey(msg tea.Msg) (screen, tea.Cmd) {
	var cmd tea.Cmd
	w.input, cmd = w.input.Update(msg)

	km, ok := msg.(tea.KeyMsg)
	if !ok {
		return w, cmd
	}
	switch km.Type {
	case tea.KeyEnter:
		key := strings.TrimSpace(w.input.Value())
		if key == "" {
			w.skip()
			return w, textinput.Blink
		}
		w.state = stateTestKey
		provider := w.currentProvider()
		return w, w.checkKeyCmd(provider, key)

	case tea.KeyTab:
		w.skip()
		return w, textinput.Blink
	}
	return w, cmd
}

func (w *wizardScreen) checkKeyCmd(provider domain.ProviderID, key string) tea.Cmd {
	return func() tea.Msg {
		p, err := providerFactory(provider, key)
		if err != nil {
			return checkDoneMsg{provider: provider, key: key, ok: false, msg: err.Error()}
		}
		checkErr := p.CheckKey(w.ctx)
		if checkErr == nil {
			return checkDoneMsg{provider: provider, key: key, ok: true, msg: "[ok] key accepted"}
		}
		if errors.Is(checkErr, errors.ErrUnsupported) {
			return checkDoneMsg{provider: provider, key: key, ok: true, msg: "[ok] key saved (provider does not support key check)"}
		}
		return checkDoneMsg{provider: provider, key: key, ok: false, msg: "[fail] " + checkErr.Error()}
	}
}

func (w *wizardScreen) updateTestKey(msg tea.Msg) (screen, tea.Cmd) {
	done, ok := msg.(checkDoneMsg)
	if !ok {
		return w, nil
	}
	w.results[done.provider] = keyResult{ok: done.ok, msg: done.msg}
	if done.ok {
		w.cfg.SetAPIKey(done.provider, done.key)
	}
	w.advance()
	return w, textinput.Blink
}

func (w *wizardScreen) updateSave(msg tea.Msg) (screen, tea.Cmd) {
	km, ok := msg.(tea.KeyMsg)
	if !ok {
		return w, nil
	}
	switch {
	case km.Type == tea.KeyEnter || km.String() == "y" || km.String() == "Y":
		w.state = stateDone
		return w, func() tea.Msg { return wizardDoneMsg{cfg: w.cfg} }
	case km.String() == "n" || km.String() == "N" || km.Type == tea.KeyEsc:
		w.state = stateDone
		return w, func() tea.Msg { return wizardDoneMsg{cfg: w.cfg, cancelled: true} }
	}
	return w, nil
}

func (w *wizardScreen) skip() {
	w.results[w.currentProvider()] = keyResult{ok: false, msg: "[skip]"}
	w.advance()
}

func (w *wizardScreen) advance() {
	w.idx++
	if w.idx >= len(w.providers) {
		w.state = stateSave
		return
	}
	w.state = stateAskKey
	w.input = newKeyInput(w.cfg.APIKeys[w.currentProvider()])
}

func (w *wizardScreen) View() string {
	var b strings.Builder
	b.WriteString(titleStyle.Render("Setup Wizard -- API Keys"))
	b.WriteString("\n\n")

	for i, p := range w.providers {
		if i >= w.idx {
			break
		}
		r := w.results[p]
		var icon string
		switch {
		case strings.HasPrefix(r.msg, "[ok]"):
			icon = successStyle.Render("[ok]")
		case strings.HasPrefix(r.msg, "[fail]"):
			icon = errorStyle.Render("[fail]")
		default:
			icon = helpStyle.Render("[skip]")
		}
		detail := strings.TrimPrefix(strings.TrimPrefix(strings.TrimPrefix(r.msg, "[ok] "), "[fail] "), "[skip]")
		b.WriteString(fmt.Sprintf("  %-12s %s %s\n", string(p), icon, strings.TrimSpace(detail)))
	}

	switch w.state {
	case stateAskKey:
		p := w.currentProvider()
		b.WriteString(fmt.Sprintf("\n  Provider %d/%d: %s\n", w.idx+1, len(w.providers), titleStyle.Render(string(p))))
		b.WriteString("  " + w.input.View() + "\n")
		b.WriteString("\n" + helpStyle.Render("  Enter = test key   Tab = skip   Ctrl+C = quit"))

	case stateTestKey:
		p := w.currentProvider()
		b.WriteString(fmt.Sprintf("\n  Testing %s ...\n", string(p)))

	case stateSave:
		keysSet := 0
		for _, p := range w.providers {
			if w.cfg.APIKeys[p] != "" {
				keysSet++
			}
		}
		b.WriteString(fmt.Sprintf("\n  %d key(s) ready to save.\n", keysSet))
		b.WriteString("\n" + helpStyle.Render("  Enter / Y = save   N / Esc = discard"))

	case stateDone:
		b.WriteString("\n  Saved.\n")
	}

	return b.String()
}

// wizardResult carries the outcome of RunWizard back through the channel.
type wizardResult struct {
	cfg ports.Config
	err error
}

// RunWizard launches the TUI setup wizard and returns the (possibly updated)
// config. The caller is responsible for persisting to disk via a ConfigStore.
func RunWizard(ctx context.Context, deps Deps) (ports.Config, error) {
	resCh := make(chan wizardResult, 1)
	ws := newWizardScreen(deps.Config)
	ws.ctx = ctx
	m := &wizardModel{ws: ws, resCh: resCh}
	prog := tea.NewProgram(m, tea.WithContext(ctx))
	if _, err := prog.Run(); err != nil {
		return deps.Config, err
	}
	r := <-resCh
	return r.cfg, r.err
}

// wizardModel adapts wizardScreen into a top-level tea.Model for RunWizard.
type wizardModel struct {
	ws    *wizardScreen
	resCh chan<- wizardResult
}

func (m *wizardModel) Init() tea.Cmd { return m.ws.Init() }

func (m *wizardModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	if km, ok := msg.(tea.KeyMsg); ok && km.Type == tea.KeyCtrlC {
		m.resCh <- wizardResult{cfg: m.ws.cfg, err: fmt.Errorf("setup cancelled")}
		return m, tea.Quit
	}
	if done, ok := msg.(wizardDoneMsg); ok {
		var err error
		if done.cancelled {
			err = fmt.Errorf("setup cancelled")
		}
		m.resCh <- wizardResult{cfg: done.cfg, err: err}
		return m, tea.Quit
	}
	next, cmd := m.ws.Update(msg)
	m.ws = next.(*wizardScreen)
	return m, cmd
}

func (m *wizardModel) View() string { return m.ws.View() }
