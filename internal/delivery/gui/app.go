package gui

import (
	"context"

	"fyne.io/fyne/v2/app"

	"github.com/leotulipan/transcribe/internal/ports"
)

// Deps holds external dependencies the GUI needs.
type Deps struct {
	Service ports.TranscribeService
	Config  ports.Config
	Logger  ports.Logger
	// SaveConfig is called when the settings window saves.
	SaveConfig func(ports.Config) error
}

// Run blocks until the user closes the window. ctx cancellation closes any
// in-flight job and ends the program loop.
func Run(ctx context.Context, deps Deps) error {
	a := app.NewWithID(appID)
	win := newMainWindow(a, ctx, deps)
	win.Show()
	a.Run()
	return nil
}
