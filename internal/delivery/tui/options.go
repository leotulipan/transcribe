package tui

import tea "github.com/charmbracelet/bubbletea"

type optionsScreen struct {
	pre  Prefill
	deps Deps
}

func newOptions(d Deps, p Prefill) *optionsScreen       { return &optionsScreen{pre: p, deps: d} }
func (o *optionsScreen) Init() tea.Cmd                   { return nil }
func (o *optionsScreen) Update(msg tea.Msg) (screen, tea.Cmd) { return o, nil }
func (o *optionsScreen) View() string                    { return "options stub — T3" }
