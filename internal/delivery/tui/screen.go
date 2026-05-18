package tui

import tea "github.com/charmbracelet/bubbletea"

// screenID identifies which sub-model is currently driving the view.
type screenID int

const (
	screenFilePicker screenID = iota
	screenOptions
	screenProgress
)

// screen is the contract every sub-model fulfills.
type screen interface {
	Init() tea.Cmd
	Update(msg tea.Msg) (screen, tea.Cmd)
	View() string
}
