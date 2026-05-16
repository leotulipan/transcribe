package gui

import (
	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/widget"
)

type settingsWindow struct{ fyne.Window }

func newSettingsWindow(parent fyne.Window, d Deps) *settingsWindow {
	a := fyne.CurrentApp()
	w := a.NewWindow(windowTitle + " — Settings")
	w.SetContent(widget.NewLabel("Settings — F3 wires this up"))
	return &settingsWindow{Window: w}
}
