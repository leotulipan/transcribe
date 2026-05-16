package gui

import (
	"context"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/widget"
)

type mainWindow struct {
	fyne.Window
}

func newMainWindow(a fyne.App, ctx context.Context, d Deps) *mainWindow {
	w := a.NewWindow(windowTitle)
	w.Resize(preferredSize)
	w.SetContent(container.NewCenter(widget.NewLabel("transcribe gui — F2 wires this up")))
	return &mainWindow{Window: w}
}
