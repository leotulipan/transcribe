package gui

import (
	"fmt"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/dialog"
	"fyne.io/fyne/v2/widget"

	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

type settingsWindow struct{ fyne.Window }

func newSettingsWindow(parent fyne.Window, d Deps) *settingsWindow {
	a := fyne.CurrentApp()
	w := a.NewWindow(windowTitle + " — Settings")
	w.Resize(fyne.NewSize(520, 420))

	cfg := d.Config

	keyEntry := func(id domain.ProviderID) *widget.Entry {
		e := widget.NewPasswordEntry()
		e.SetText(cfg.APIKeys[id])
		return e
	}
	groq     := keyEntry(domain.ProviderGroq)
	openai   := keyEntry(domain.ProviderOpenAI)
	assembly := keyEntry(domain.ProviderAssemblyAI)
	eleven   := keyEntry(domain.ProviderElevenLabs)
	gemini   := keyEntry(domain.ProviderGemini)
	mistral  := keyEntry(domain.ProviderMistral)

	ffmpegEntry := widget.NewEntry()
	ffmpegEntry.SetText(cfg.FFmpegPath)
	ffmpegEntry.SetPlaceHolder("auto-discover via PATH")

	langEntry := widget.NewEntry()
	langEntry.SetText(cfg.DefaultLanguage)
	langEntry.SetPlaceHolder("en, de, fr, ...")

	save := widget.NewButton("Save", func() {
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
			FFmpegPath:      ffmpegEntry.Text,
		}
		if d.SaveConfig == nil {
			dialog.ShowInformation("Save", "(no SaveConfig wired)", w)
			return
		}
		if err := d.SaveConfig(next); err != nil {
			dialog.ShowError(err, w)
			return
		}
		dialog.ShowInformation("Saved",
			fmt.Sprintf("Config written. Restart transcribe to pick up new keys."), w)
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
	w.SetContent(container.NewBorder(nil, container.NewHBox(save, cancel), nil, nil, form))
	return &settingsWindow{Window: w}
}
