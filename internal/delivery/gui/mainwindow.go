package gui

import (
	"context"
	"fmt"
	"strings"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/dialog"
	"fyne.io/fyne/v2/widget"

	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

type mainWindow struct {
	fyne.Window
	deps Deps
	ctx  context.Context

	pathEntry  *widget.Entry
	provider   *widget.Select
	model      *widget.Select
	language   *widget.Entry
	fmtText    *widget.Check
	fmtSRT     *widget.Check
	fmtDavinci *widget.Check

	bar         *widget.ProgressBarInfinite
	determinate *widget.ProgressBar
	logArea     *widget.Entry
	startBtn    *widget.Button
	cancelBtn   *widget.Button

	currentJob ports.Job
}

func newMainWindow(a fyne.App, ctx context.Context, d Deps) *mainWindow {
	w := a.NewWindow(windowTitle)
	w.Resize(preferredSize)

	m := &mainWindow{Window: w, deps: d, ctx: ctx}

	// File row
	m.pathEntry = widget.NewEntry()
	m.pathEntry.SetPlaceHolder("Pick an audio or video file")
	browse := widget.NewButton("Browse…", m.onBrowse)

	// Provider + model row
	var providerIDs []string
	for _, id := range d.Service.ListProviders() {
		providerIDs = append(providerIDs, string(id))
	}
	m.provider = widget.NewSelect(providerIDs, m.onProviderChanged)
	m.model = widget.NewSelect(nil, nil)
	if len(providerIDs) > 0 {
		m.provider.SetSelected(providerIDs[0])
	}

	// Language + formats
	m.language = widget.NewEntry()
	m.language.SetPlaceHolder("language (blank = auto)")
	m.language.SetText(d.Config.DefaultLanguage)
	m.fmtText = widget.NewCheck("text", nil)
	m.fmtText.SetChecked(true)
	m.fmtSRT = widget.NewCheck("srt", nil)
	m.fmtDavinci = widget.NewCheck("davinci_srt", nil)

	// Progress + log
	m.bar = widget.NewProgressBarInfinite()
	m.bar.Hide()
	m.determinate = widget.NewProgressBar()
	m.determinate.Hide()
	m.logArea = widget.NewMultiLineEntry()
	m.logArea.SetMinRowsVisible(8)
	m.logArea.Disable()

	// Actions
	m.startBtn = widget.NewButton("Start", m.onStart)
	m.cancelBtn = widget.NewButton("Cancel", m.onCancel)
	m.cancelBtn.Disable()

	settingsBtn := widget.NewButton("Settings…", m.onSettings)

	layout := container.NewVBox(
		widget.NewLabelWithStyle("File", fyne.TextAlignLeading, fyne.TextStyle{Bold: true}),
		container.NewBorder(nil, nil, nil, browse, m.pathEntry),

		widget.NewLabelWithStyle("Provider", fyne.TextAlignLeading, fyne.TextStyle{Bold: true}),
		container.NewGridWithColumns(2, m.provider, m.model),

		widget.NewLabelWithStyle("Language", fyne.TextAlignLeading, fyne.TextStyle{Bold: true}),
		m.language,

		widget.NewLabelWithStyle("Output formats", fyne.TextAlignLeading, fyne.TextStyle{Bold: true}),
		container.NewHBox(m.fmtText, m.fmtSRT, m.fmtDavinci),

		widget.NewLabelWithStyle("Progress", fyne.TextAlignLeading, fyne.TextStyle{Bold: true}),
		m.bar, m.determinate, m.logArea,

		container.NewHBox(m.startBtn, m.cancelBtn, settingsBtn),
	)

	w.SetContent(container.NewPadded(layout))
	w.SetCloseIntercept(m.onWindowClose)

	// Trigger initial model list
	if len(providerIDs) > 0 {
		m.onProviderChanged(providerIDs[0])
	}
	return m
}

func (m *mainWindow) onBrowse() {
	dialog.ShowFileOpen(func(rc fyne.URIReadCloser, err error) {
		if err != nil || rc == nil {
			return
		}
		defer rc.Close()
		m.pathEntry.SetText(rc.URI().Path())
	}, m.Window)
}

func (m *mainWindow) onProviderChanged(id string) {
	if id == "" {
		m.model.Options = nil
		m.model.Refresh()
		return
	}
	models, err := m.deps.Service.ListModels(domain.ProviderID(id))
	if err != nil {
		m.logf("provider %s: %v", id, err)
		return
	}
	m.model.Options = models
	if len(models) > 0 {
		m.model.SetSelected(models[0])
	}
	m.model.Refresh()
}

func (m *mainWindow) onStart() {
	formats := m.selectedFormats()
	if len(formats) == 0 {
		dialog.ShowInformation("Pick a format", "Select at least one output format.", m.Window)
		return
	}
	if m.pathEntry.Text == "" {
		dialog.ShowInformation("Pick a file", "Choose an audio or video file first.", m.Window)
		return
	}
	if m.provider.Selected == "" {
		dialog.ShowInformation("Pick a provider", "Select a provider (run Settings if none are listed).", m.Window)
		return
	}
	m.lockUI(true)
	m.bar.Show()
	m.bar.Start()
	m.determinate.SetValue(0)
	m.determinate.Hide()
	m.logArea.SetText("")

	req := domain.Request{
		InputPath: m.pathEntry.Text,
		Provider:  domain.ProviderID(m.provider.Selected),
		Model:     m.model.Selected,
		Language:  strings.TrimSpace(m.language.Text),
		Formats:   formats,
		UseCache:  true,
	}
	if m.fmtDavinci.Checked {
		req.DaVinciOpts = &domain.DaVinciOptions{}
	}

	job, err := runJob(m.ctx, m.deps.Service, req,
		m.onProgress, m.onDone,
	)
	if err != nil {
		m.lockUI(false)
		m.bar.Hide()
		dialog.ShowError(err, m.Window)
		return
	}
	m.currentJob = job
}

func (m *mainWindow) selectedFormats() []domain.OutputFormat {
	var out []domain.OutputFormat
	if m.fmtText.Checked {
		out = append(out, domain.FormatText)
	}
	if m.fmtSRT.Checked {
		out = append(out, domain.FormatSRT)
	}
	if m.fmtDavinci.Checked {
		out = append(out, domain.FormatDavinciSRT)
	}
	return out
}

func (m *mainWindow) onProgress(ev domain.ProgressEvent) {
	m.logf("[%s] %s", ev.Stage, ev.Message)
	if ev.Percent >= 0 {
		m.bar.Hide()
		m.determinate.Show()
		m.determinate.SetValue(ev.Percent)
	}
}

func (m *mainWindow) onDone(res *domain.Result, err error) {
	m.lockUI(false)
	m.bar.Hide()
	m.determinate.Hide()
	m.currentJob = nil
	if err != nil {
		dialog.ShowError(err, m.Window)
		return
	}
	preview := res.Text
	if len(preview) > 300 {
		preview = preview[:300] + "…"
	}
	dialog.ShowInformation("Done", preview, m.Window)
}

func (m *mainWindow) onCancel() {
	if m.currentJob != nil {
		m.currentJob.Cancel()
	}
}

func (m *mainWindow) onSettings() {
	newSettingsWindow(m.Window, m.deps).Show()
}

func (m *mainWindow) onWindowClose() {
	if m.currentJob != nil {
		dialog.ShowConfirm("Cancel running job?", "A transcription is in progress. Close anyway?",
			func(ok bool) {
				if ok {
					m.currentJob.Cancel()
					m.Window.Close()
				}
			}, m.Window)
		return
	}
	m.Window.Close()
}

func (m *mainWindow) lockUI(lock bool) {
	set := func(en bool, btns ...fyne.Disableable) {
		for _, b := range btns {
			if en {
				b.Enable()
			} else {
				b.Disable()
			}
		}
	}
	set(!lock, m.startBtn, m.provider, m.model, m.fmtText, m.fmtSRT, m.fmtDavinci)
	set(lock, m.cancelBtn)
	if lock {
		m.pathEntry.Disable()
		m.language.Disable()
	} else {
		m.pathEntry.Enable()
		m.language.Enable()
	}
}

func (m *mainWindow) logf(format string, args ...any) {
	m.logArea.SetText(m.logArea.Text + fmt.Sprintf(format+"\n", args...))
}
