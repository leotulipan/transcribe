package gui

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/dialog"
	"fyne.io/fyne/v2/widget"

	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/core/services"
	"github.com/leotulipan/transcribe/internal/ports"
)

type mainWindow struct {
	fyne.Window
	deps *Deps
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

	// Batch state. When the input path is a directory we enumerate it at
	// Start time and process files sequentially. batchFiles is reset to
	// nil between runs.
	batchFiles  []string
	batchIndex  int
	batchCancel bool
}

func newMainWindow(a fyne.App, ctx context.Context, d *Deps) *mainWindow {
	w := a.NewWindow(windowTitle)
	w.Resize(preferredSize)

	m := &mainWindow{Window: w, deps: d, ctx: ctx}

	// File row
	m.pathEntry = widget.NewEntry()
	m.pathEntry.SetPlaceHolder("Pick a file or folder (or drop one here)")
	browseFile := widget.NewButton("File…", m.onBrowseFile)
	browseDir := widget.NewButton("Folder…", m.onBrowseFolder)

	// Accept drag-and-drop of files / folders onto the window. Desktop-only;
	// Fyne's SetOnDropped is a no-op on mobile, which is fine.
	w.SetOnDropped(func(_ fyne.Position, uris []fyne.URI) {
		if len(uris) == 0 {
			return
		}
		m.pathEntry.SetText(uris[0].Path())
	})

	// Provider + model row
	svc := d.Service()
	var providerIDs []string
	for _, id := range svc.ListProviders() {
		providerIDs = append(providerIDs, string(id))
	}
	m.provider = widget.NewSelect(providerIDs, m.onProviderChanged)
	m.model = widget.NewSelect(nil, nil)
	if len(providerIDs) > 0 {
		m.provider.SetSelected(providerIDs[0])
	}
	refreshModels := widget.NewButton("↻", m.onRefreshModels)
	refreshModels.SetIcon(nil) // text-only; emoji renders consistently in Fyne

	// Language + formats
	m.language = widget.NewEntry()
	m.language.SetPlaceHolder("language (blank = auto)")
	m.language.SetText(d.Config().DefaultLanguage)
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
		widget.NewLabelWithStyle("File or folder", fyne.TextAlignLeading, fyne.TextStyle{Bold: true}),
		container.NewBorder(nil, nil, nil, container.NewHBox(browseFile, browseDir), m.pathEntry),

		widget.NewLabelWithStyle("Provider", fyne.TextAlignLeading, fyne.TextStyle{Bold: true}),
		container.NewBorder(nil, nil, nil, refreshModels,
			container.NewGridWithColumns(2, m.provider, m.model),
		),

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

func (m *mainWindow) onBrowseFile() {
	dialog.ShowFileOpen(func(rc fyne.URIReadCloser, err error) {
		if err != nil || rc == nil {
			return
		}
		defer rc.Close()
		m.pathEntry.SetText(rc.URI().Path())
	}, m.Window)
}

// onBrowseFolder opens a folder picker. The chosen path is stored verbatim
// in the entry; enumeration is deferred to Start so the user can still edit
// the path or back out.
func (m *mainWindow) onBrowseFolder() {
	dialog.ShowFolderOpen(func(uri fyne.ListableURI, err error) {
		if err != nil || uri == nil {
			return
		}
		m.pathEntry.SetText(uri.Path())
	}, m.Window)
}

func (m *mainWindow) onProviderChanged(id string) {
	if id == "" {
		m.model.Options = nil
		m.model.Refresh()
		return
	}
	models, err := m.deps.Service().ListModels(domain.ProviderID(id))
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
		dialog.ShowInformation("Pick a file", "Choose an audio file, video file, or folder first.", m.Window)
		return
	}
	if m.provider.Selected == "" {
		dialog.ShowInformation("Pick a provider", "Select a provider (run Settings if none are listed).", m.Window)
		return
	}

	// Resolve the entry into a list of files. Single file → [file].
	// Directory → walk + filter by AudioExtensions. Done at Start so the
	// user can still edit the path after picking a folder.
	files, err := services.EnumerateAudioFiles(m.pathEntry.Text)
	if err != nil {
		dialog.ShowError(fmt.Errorf("read %s: %w", m.pathEntry.Text, err), m.Window)
		return
	}
	if len(files) == 0 {
		dialog.ShowInformation("No files",
			"No audio or video files found in that folder.", m.Window)
		return
	}

	m.lockUI(true)
	m.logArea.SetText("")
	m.batchFiles = files
	m.batchIndex = 0
	m.batchCancel = false

	if len(files) > 1 {
		m.logf("batch: %d files to process", len(files))
	}
	m.startNextJob(formats)
}

// startNextJob submits the file at m.batchIndex and arranges for the next
// one to start when it finishes. Called serially; never concurrently.
func (m *mainWindow) startNextJob(formats []domain.OutputFormat) {
	if m.batchCancel || m.batchIndex >= len(m.batchFiles) {
		m.finishBatch(nil)
		return
	}
	file := m.batchFiles[m.batchIndex]

	m.bar.Show()
	m.bar.Start()
	m.determinate.SetValue(0)
	m.determinate.Hide()

	if len(m.batchFiles) > 1 {
		m.logf("[%d/%d] %s", m.batchIndex+1, len(m.batchFiles), filepathBase(file))
	}

	req := domain.Request{
		InputPath: file,
		Provider:  domain.ProviderID(m.provider.Selected),
		Model:     m.model.Selected,
		Language:  strings.TrimSpace(m.language.Text),
		Formats:   formats,
		UseCache:  true,
	}
	if m.fmtDavinci.Checked {
		req.DaVinciOpts = &domain.DaVinciOptions{}
	}

	job, err := runJob(m.ctx, m.deps.Service(), req,
		m.onProgress,
		func(res *domain.Result, err error) {
			// On per-file error, abort the rest of the batch and surface
			// the error. The user can fix and re-run.
			if err != nil {
				m.finishBatch(err)
				return
			}
			m.batchIndex++
			if m.batchIndex < len(m.batchFiles) && !m.batchCancel {
				m.startNextJob(formats)
				return
			}
			m.finishBatch(nil)
		},
	)
	if err != nil {
		m.finishBatch(err)
		return
	}
	m.currentJob = job
}

func (m *mainWindow) finishBatch(err error) {
	m.lockUI(false)
	m.bar.Hide()
	m.determinate.Hide()
	m.currentJob = nil
	m.batchFiles = nil
	m.batchIndex = 0
	if err != nil {
		dialog.ShowError(err, m.Window)
		return
	}
	if m.batchCancel {
		m.logf("batch cancelled")
		return
	}
	dialog.ShowInformation("Done", "All files processed.", m.Window)
}

// filepathBase is a tiny helper to keep mainwindow.go from importing path/filepath
// just for one call. Mirrors filepath.Base.
func filepathBase(p string) string {
	for i := len(p) - 1; i >= 0; i-- {
		if p[i] == '/' || p[i] == os.PathSeparator {
			return p[i+1:]
		}
	}
	return p
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

// onDone is retained for the single-file legacy path but the batch loop
// now goes through startNextJob's inline callback. Kept to avoid breaking
// anything in tests that might still reference it.
func (m *mainWindow) onDone(res *domain.Result, err error) {
	m.lockUI(false)
	m.bar.Hide()
	m.determinate.Hide()
	m.currentJob = nil
	if err != nil {
		dialog.ShowError(err, m.Window)
		return
	}
	preview := ""
	if res != nil {
		preview = res.Text
	}
	if len(preview) > 300 {
		preview = preview[:300] + "…"
	}
	dialog.ShowInformation("Done", preview, m.Window)
}

func (m *mainWindow) onCancel() {
	m.batchCancel = true
	if m.currentJob != nil {
		m.currentJob.Cancel()
	}
}

// onRefreshModels calls the current provider's DiscoverModels endpoint,
// persists the result via SaveConfig + Reload, and refreshes the dropdown.
// Per-plan decision: per-provider button (no all-at-once action).
func (m *mainWindow) onRefreshModels() {
	if m.provider.Selected == "" {
		dialog.ShowInformation("Pick a provider",
			"Select a provider first.", m.Window)
		return
	}
	pid := domain.ProviderID(m.provider.Selected)
	go func() {
		ctx, cancel := context.WithTimeout(m.ctx, 30*time.Second)
		defer cancel()
		svc := m.deps.Service()
		models, err := svc.DiscoverModels(ctx, pid)
		fyne.Do(func() {
			if err != nil {
				dialog.ShowError(fmt.Errorf("discover %s: %w", pid, err), m.Window)
				return
			}
			// Persist into config.
			cfg := m.deps.Config()
			if cfg.DiscoveredModels == nil {
				cfg.DiscoveredModels = map[domain.ProviderID][]string{}
			}
			cfg.DiscoveredModels[pid] = models
			if m.deps.SaveConfig != nil {
				if saveErr := m.deps.SaveConfig(cfg); saveErr != nil {
					dialog.ShowError(saveErr, m.Window)
					return
				}
				if reloadErr := m.deps.Reload(); reloadErr != nil {
					dialog.ShowError(reloadErr, m.Window)
					return
				}
			}
			m.model.Options = models
			if len(models) > 0 {
				m.model.SetSelected(models[0])
			}
			m.model.Refresh()
			m.logf("refreshed %s: %d models", pid, len(models))
		})
	}()
}

func (m *mainWindow) onSettings() {
	w := newSettingsWindow(m.Window, m.deps, m.refreshProviders)
	w.Show()
}

// refreshProviders re-reads the provider list from the (possibly newly
// reloaded) service and rebinds the dropdowns. Called after Settings saves.
func (m *mainWindow) refreshProviders() {
	svc := m.deps.Service()
	var ids []string
	for _, id := range svc.ListProviders() {
		ids = append(ids, string(id))
	}
	m.provider.Options = ids
	if len(ids) > 0 {
		// Preserve current selection if still valid, else pick first.
		current := m.provider.Selected
		valid := false
		for _, id := range ids {
			if id == current {
				valid = true
				break
			}
		}
		if !valid {
			m.provider.SetSelected(ids[0])
		} else {
			m.onProviderChanged(current)
		}
	} else {
		m.provider.ClearSelected()
		m.model.Options = nil
		m.model.Refresh()
	}
	m.provider.Refresh()
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
