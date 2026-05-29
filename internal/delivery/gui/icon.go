package gui

import (
	_ "embed"

	"fyne.io/fyne/v2"
)

// Fyne's window icon (top-left chrome) and the OS taskbar icon are separate
// things on Windows. The taskbar icon is baked in by the Windows resource
// file (cmd/transcribe-gui/resource.syso); Fyne needs an explicit
// fyne.Resource set via App.SetIcon to render the window chrome icon.
// We embed the PNG here (.ico is not a valid Fyne resource).

//go:embed assets/icon.png
var iconPNG []byte

var appIcon = fyne.NewStaticResource("transcribe.png", iconPNG)
