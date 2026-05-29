package gui

import (
	"fmt"
	"net/url"
	"os"
	"path/filepath"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/dialog"
)

// readmeFallbackURL is the public source-of-truth README, used when the
// local copy installed alongside the binary cannot be located.
const readmeFallbackURL = "https://github.com/leotulipan/transcribe/blob/main/README.md"

// onReadme opens README.md next to the running binary (where the Windows
// installer drops it) using the OS default handler. Falls back to the
// GitHub URL when the local file is missing — e.g. when running a `go run`
// dev build out of the repo without an install step.
func (m *mainWindow) onReadme() {
	app := fyne.CurrentApp()
	if path, ok := localReadmePath(); ok {
		u := &url.URL{Scheme: "file", Path: filepath.ToSlash(path)}
		if err := app.OpenURL(u); err == nil {
			return
		}
		// fall through to remote on OpenURL failure
	}
	u, err := url.Parse(readmeFallbackURL)
	if err != nil {
		dialog.ShowError(fmt.Errorf("open readme: %w", err), m.Window)
		return
	}
	if err := app.OpenURL(u); err != nil {
		dialog.ShowError(fmt.Errorf("open readme: %w", err), m.Window)
	}
}

// localReadmePath returns the absolute path to README.md sitting next to
// the executable, or ok=false when no readable file is found. The Windows
// installer drops README.md into {app}; dev builds normally don't.
func localReadmePath() (string, bool) {
	exe, err := os.Executable()
	if err != nil {
		return "", false
	}
	candidate := filepath.Join(filepath.Dir(exe), "README.md")
	if info, err := os.Stat(candidate); err == nil && !info.IsDir() {
		return candidate, true
	}
	return "", false
}
