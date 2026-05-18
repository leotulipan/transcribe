package gui

import (
	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/dialog"
	"fyne.io/fyne/v2/widget"

	"github.com/leotulipan/transcribe/internal/adapters/audio"
	configadapter "github.com/leotulipan/transcribe/internal/adapters/config"
	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

type settingsWindow struct{ fyne.Window }

// newSettingsWindow opens the settings dialog. After a successful save,
// onSaved (if non-nil) is called on the Fyne main thread so the main window
// can refresh anything that depends on the new config (e.g. provider list).
func newSettingsWindow(parent fyne.Window, d *Deps, onSaved func()) *settingsWindow {
	a := fyne.CurrentApp()
	w := a.NewWindow(windowTitle + " — Settings")
	w.Resize(fyne.NewSize(560, 460))

	// Always pre-populate from the latest in-memory config (which reflects
	// the most recent save, since Reload swaps d.config). Falls back to the
	// startup snapshot if nothing has been reloaded yet.
	cfg := d.Config()

	keyEntry := func(id domain.ProviderID) *widget.Entry {
		e := widget.NewPasswordEntry()
		e.SetText(cfg.APIKeys[id])
		return e
	}
	groq := keyEntry(domain.ProviderGroq)
	openai := keyEntry(domain.ProviderOpenAI)
	assembly := keyEntry(domain.ProviderAssemblyAI)
	eleven := keyEntry(domain.ProviderElevenLabs)
	gemini := keyEntry(domain.ProviderGemini)
	mistral := keyEntry(domain.ProviderMistral)

	ffmpegEntry := widget.NewEntry()
	ffmpegEntry.SetText(cfg.FFmpegPath)
	ffmpegEntry.SetPlaceHolder("auto-discover via PATH")

	langEntry := widget.NewEntry()
	langEntry.SetText(cfg.DefaultLanguage)
	langEntry.SetPlaceHolder("en, de, fr, ...")

	configPathLabel := widget.NewLabel("Config file: " + configadapter.New().Path())
	configPathLabel.Wrapping = fyne.TextWrapBreak
	configPathLabel.TextStyle = fyne.TextStyle{Italic: true}

	save := widget.NewButton("Save", func() {
		ffmpegResolved := ffmpegEntry.Text
		if ffmpegResolved != "" {
			r, err := audio.ResolveBinary(ffmpegResolved, "ffmpeg")
			if err != nil {
				dialog.ShowError(err, w)
				return
			}
			ffmpegResolved = r
			ffmpegEntry.SetText(r) // surface canonical path back to the user
		}
		next := ports.Config{
			APIKeys: map[domain.ProviderID]string{
				domain.ProviderGroq:       groq.Text,
				domain.ProviderOpenAI:     openai.Text,
				domain.ProviderAssemblyAI: assembly.Text,
				domain.ProviderElevenLabs: eleven.Text,
				domain.ProviderGemini:     gemini.Text,
				domain.ProviderMistral:    mistral.Text,
			},
			DefaultProvider: cfg.DefaultProvider,
			DefaultLanguage: langEntry.Text,
			FFmpegPath:      ffmpegResolved,
		}
		if d.SaveConfig == nil {
			dialog.ShowInformation("Save", "(no SaveConfig wired)", w)
			return
		}
		if err := d.SaveConfig(next); err != nil {
			dialog.ShowError(err, w)
			return
		}
		// Hot-swap providers/service so the user doesn't have to restart.
		if err := d.Reload(); err != nil {
			dialog.ShowError(err, w)
			return
		}
		if onSaved != nil {
			onSaved()
		}
		dialog.ShowInformation("Saved", "Settings applied. No restart needed.", w)
	})

	cancel := widget.NewButton("Close", func() { w.Close() })

	form := widget.NewForm(
		widget.NewFormItem("Groq", groq),
		widget.NewFormItem("OpenAI", openai),
		widget.NewFormItem("AssemblyAI", assembly),
		widget.NewFormItem("ElevenLabs", eleven),
		widget.NewFormItem("Gemini", gemini),
		widget.NewFormItem("Mistral", mistral),
		widget.NewFormItem("FFmpeg path", ffmpegEntry),
		widget.NewFormItem("Default language", langEntry),
	)
	content := container.NewBorder(
		configPathLabel,
		container.NewHBox(save, cancel),
		nil, nil,
		form,
	)
	w.SetContent(content)
	return &settingsWindow{Window: w}
}
