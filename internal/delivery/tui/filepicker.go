package tui

import tea "github.com/charmbracelet/bubbletea"

type filePicker struct{ pre Prefill }

func newFilePicker(p Prefill) *filePicker             { return &filePicker{pre: p} }
func (f *filePicker) Init() tea.Cmd                   { return nil }
func (f *filePicker) Update(msg tea.Msg) (screen, tea.Cmd) { return f, nil }
func (f *filePicker) View() string                    { return "filepicker stub — T2" }
