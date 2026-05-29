package gui

import (
	"net/url"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/dialog"
	"fyne.io/fyne/v2/widget"
)

// onAbout shows a simple modal with the build version, author, project
// links, and license. Version comes from deps.Version (injected via
// -ldflags by the installer build). Hyperlinks open in the user's default
// browser via Fyne's URI handling.
func (m *mainWindow) onAbout() {
	content := container.NewVBox(
		widget.NewLabelWithStyle("Audio Transcribe",
			fyne.TextAlignCenter, fyne.TextStyle{Bold: true}),
		widget.NewLabel("Version "+m.deps.Version),
		widget.NewLabel("by Leonard Tulipan"),
		widget.NewHyperlink("leotulipan.at", mustParseURL("https://leotulipan.at")),
		widget.NewHyperlink("github.com/leotulipan/transcribe",
			mustParseURL("https://github.com/leotulipan/transcribe")),
		widget.NewLabel("MIT License"),
	)
	dialog.ShowCustom("About", "Close", content, m.Window)
}

// mustParseURL is a tiny helper for compile-time-known URLs used in the
// About / Settings dialogs. Returns nil on parse error so Fyne renders the
// hyperlink as plain text rather than panicking.
func mustParseURL(raw string) *url.URL {
	u, err := url.Parse(raw)
	if err != nil {
		return nil
	}
	return u
}
