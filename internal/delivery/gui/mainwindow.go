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
	outputDir  *widget.Entry
	fmtText    *widget.Check
	fmtSRT     *widget.Check
	fmtWordSRT *widget.Check
	fmtDavinci *widget.Check

	// Advanced — subtitle wrapping
	charsPerLine     *widget.Entry
	wordsPerSubtitle *widget.Entry
	startHour        *widget.Entry

	// Advanced — diarization
	diarize       *widget.Check
	speakerLabels *widget.Check
	numSpeakers   *widget.Entry

	// Advanced — DaVinci timing
	paddingStartMs *widget.Entry
	paddingEndMs   *widget.Entry
	silentMs       *widget.Entry
	fps            *widget.Entry
	fpsOffsetStart *widget.Entry
	fpsOffsetEnd   *widget.Entry
	showPauses     *widget.Check

	// Advanced — fillers
	removeFillers *widget.Check
	fillerLines   *widget.Check
	fillerWords   *widget.Entry

	// Advanced — audio pipeline
	sizeThresholdMB *widget.Entry
	chunkLengthSec  *widget.Entry
	overlapSec      *widget.Entry
	useInput        *widget.Check
	usePCM          *widget.Check
	keep            *widget.Check
	keepFLAC        *widget.Check

	// Advanced — I/O
	force           *widget.Check
	saveCleanedJSON *widget.Check
	extensions      *widget.Entry

	// Advanced — provider hints
	keyTermsPrompt *widget.Entry
	speechModels   *widget.Entry

	bar         *widget.ProgressBarInfinite
	determinate *widget.ProgressBar
	logRich     *widget.RichText
	logScroll   *container.Scroll
	logBuf      string
	startBtn    *widget.Button
	cancelBtn   *widget.Button
	// Mirror buttons pinned to the top toolbar so Start/Cancel stay
	// reachable even when the Advanced accordion has pushed the bottom row
	// out of view. Kept in sync with startBtn/cancelBtn by lockUI.
	topStartBtn  *widget.Button
	topCancelBtn *widget.Button

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
	if initial := preferredProvider(providerIDs); initial != "" {
		m.provider.SetSelected(initial)
	}
	refreshModels := widget.NewButton("↻", m.onRefreshModels)
	refreshModels.SetIcon(nil) // text-only; emoji renders consistently in Fyne

	// Language + formats
	m.language = widget.NewEntry()
	m.language.SetPlaceHolder("language (blank = auto)")
	m.language.SetText(d.Config().DefaultLanguage)
	m.outputDir = widget.NewEntry()
	m.outputDir.SetPlaceHolder("output directory (blank = next to input)")
	m.fmtText = widget.NewCheck("text", nil)
	m.fmtText.SetChecked(true)
	m.fmtSRT = widget.NewCheck("srt", nil)
	m.fmtWordSRT = widget.NewCheck("word_srt", nil)
	m.fmtDavinci = widget.NewCheck("davinci_srt", nil)

	// Advanced widgets. All entries are plain text; runtime parses them into
	// the right type and falls back to the default if the user clears the
	// field. This keeps the UI simple at the cost of one bit of validation.
	m.charsPerLine = numEntry("0 = no wrap")
	m.wordsPerSubtitle = numEntry("0 = default 7 (DaVinci)")
	m.startHour = numEntry("0")
	m.diarize = widget.NewCheck("diarize (request speaker IDs from provider)", nil)
	m.speakerLabels = widget.NewCheck("prefix subtitles with [Speaker X]:", nil)
	m.numSpeakers = numEntry("0 = unset; 1..32")
	m.paddingStartMs = numEntry("ms (shift starts earlier)")
	m.paddingEndMs = numEntry("ms (shrink ends earlier)")
	m.silentMs = numEntry("1500")
	m.silentMs.SetText("1500")
	m.fps = numEntry("0 = no snapping (e.g. 23.976)")
	m.fpsOffsetStart = numEntry("-1 (one frame earlier)")
	m.fpsOffsetStart.SetText("-1")
	m.fpsOffsetEnd = numEntry("0")
	m.showPauses = widget.NewCheck("show (...) pause markers (DaVinci)", nil)
	m.showPauses.SetChecked(true)
	m.removeFillers = widget.NewCheck("drop filler words from output", nil)
	m.fillerLines = widget.NewCheck("uppercase fillers (own line in DaVinci)", nil)
	m.fillerLines.SetChecked(true)
	m.fillerWords = widget.NewEntry()
	m.fillerWords.SetPlaceHolder("um,uh,ähm,äh,hm,hmm (blank = default)")
	m.sizeThresholdMB = numEntry("100 (MB)")
	m.sizeThresholdMB.SetText("100")
	m.chunkLengthSec = numEntry("0 = derive from byte budget")
	m.overlapSec = numEntry("0 = no overlap")
	m.useInput = widget.NewCheck("send original file as-is (no conversion)", nil)
	m.usePCM = widget.NewCheck("convert to PCM WAV instead of preferred codec", nil)
	m.keep = widget.NewCheck("retain all intermediate files", nil)
	m.keepFLAC = widget.NewCheck("retain FLAC intermediates", nil)
	m.force = widget.NewCheck("force re-transcribe (ignore cached sidecar)", nil)
	m.saveCleanedJSON = widget.NewCheck("save normalized sidecar JSON", nil)
	m.extensions = widget.NewEntry()
	m.extensions.SetPlaceHolder("mp3,m4a (filter for folder enumeration)")
	m.keyTermsPrompt = widget.NewEntry()
	m.keyTermsPrompt.SetPlaceHolder("term1,term2 (assemblyai)")
	m.speechModels = widget.NewEntry()
	m.speechModels.SetPlaceHolder("e.g. universal-3-pro,universal-2 — tried in order, falls back on language mismatch (assemblyai)")

	// Progress + log
	m.bar = widget.NewProgressBarInfinite()
	m.bar.Hide()
	m.determinate = widget.NewProgressBar()
	m.determinate.Hide()
	// Read-only activity log. A disabled Entry renders its text in Fyne's
	// muted "disabled" color (grey-on-grey, hard to read); RichText renders at
	// full foreground contrast and is read-only by nature.
	m.logRich = widget.NewRichTextWithText("")
	m.logRich.Wrapping = fyne.TextWrapWord
	m.logScroll = container.NewVScroll(m.logRich)
	m.logScroll.SetMinSize(fyne.NewSize(0, 160))

	// Actions — bottom row (legacy position, kept so users with muscle
	// memory still find it under the scroll area).
	m.startBtn = widget.NewButton("Start", m.onStart)
	m.cancelBtn = widget.NewButton("Cancel", m.onCancel)
	m.cancelBtn.Disable()

	settingsBtn := widget.NewButton("Settings…", m.onSettings)

	// Top toolbar mirrors. Fyne won't let one widget live in two parents,
	// so we instantiate fresh buttons that share the same handlers; lockUI
	// disables / enables both in lockstep.
	m.topStartBtn = widget.NewButton("Start", m.onStart)
	m.topStartBtn.Importance = widget.HighImportance
	m.topCancelBtn = widget.NewButton("Cancel", m.onCancel)
	m.topCancelBtn.Disable()
	topSettingsBtn := widget.NewButton("Settings…", m.onSettings)
	readmeBtn := widget.NewButton("Readme", m.onReadme)
	aboutBtn := widget.NewButton("About…", m.onAbout)

	advanced := widget.NewAccordion(
		widget.NewAccordionItem("Subtitle wrapping",
			widget.NewForm(
				widget.NewFormItem("Chars per line", m.charsPerLine),
				widget.NewFormItem("Words per subtitle", m.wordsPerSubtitle),
				widget.NewFormItem("Start hour", m.startHour),
			),
		),
		widget.NewAccordionItem("Diarization",
			container.NewVBox(m.diarize, m.speakerLabels,
				widget.NewForm(widget.NewFormItem("Num speakers", m.numSpeakers)),
			),
		),
		widget.NewAccordionItem("DaVinci timing",
			container.NewVBox(m.showPauses,
				widget.NewForm(
					widget.NewFormItem("Padding start (ms)", m.paddingStartMs),
					widget.NewFormItem("Padding end (ms)", m.paddingEndMs),
					widget.NewFormItem("Silent threshold (ms)", m.silentMs),
					widget.NewFormItem("FPS", m.fps),
					widget.NewFormItem("FPS offset start (frames)", m.fpsOffsetStart),
					widget.NewFormItem("FPS offset end (frames)", m.fpsOffsetEnd),
				),
			),
		),
		widget.NewAccordionItem("Filler words",
			container.NewVBox(m.removeFillers, m.fillerLines,
				widget.NewForm(widget.NewFormItem("Filler words (csv)", m.fillerWords)),
			),
		),
		widget.NewAccordionItem("Audio pipeline",
			container.NewVBox(m.useInput, m.usePCM, m.keep, m.keepFLAC,
				widget.NewForm(
					widget.NewFormItem("Size threshold (MB)", m.sizeThresholdMB),
					widget.NewFormItem("Chunk length (s)", m.chunkLengthSec),
					widget.NewFormItem("Overlap (s)", m.overlapSec),
				),
			),
		),
		widget.NewAccordionItem("I/O & workflow",
			container.NewVBox(m.force, m.saveCleanedJSON,
				widget.NewForm(
					widget.NewFormItem("Output dir", m.outputDir),
					widget.NewFormItem("Extensions (csv)", m.extensions),
				),
			),
		),
		widget.NewAccordionItem("Provider hints (assemblyai)",
			widget.NewForm(
				widget.NewFormItem("Key terms (csv)", m.keyTermsPrompt),
				widget.NewFormItem("Speech models (csv)", m.speechModels),
			),
		),
	)

	form := container.NewVBox(
		widget.NewLabelWithStyle("File or folder", fyne.TextAlignLeading, fyne.TextStyle{Bold: true}),
		container.NewBorder(nil, nil, nil, container.NewHBox(browseFile, browseDir), m.pathEntry),

		widget.NewLabelWithStyle("Provider", fyne.TextAlignLeading, fyne.TextStyle{Bold: true}),
		container.NewBorder(nil, nil, nil, refreshModels,
			container.NewGridWithColumns(2, m.provider, m.model),
		),

		widget.NewLabelWithStyle("Language", fyne.TextAlignLeading, fyne.TextStyle{Bold: true}),
		m.language,

		widget.NewLabelWithStyle("Output formats", fyne.TextAlignLeading, fyne.TextStyle{Bold: true}),
		container.NewHBox(m.fmtText, m.fmtSRT, m.fmtWordSRT, m.fmtDavinci),

		widget.NewLabelWithStyle("Advanced", fyne.TextAlignLeading, fyne.TextStyle{Bold: true}),
		advanced,

		widget.NewLabelWithStyle("Progress", fyne.TextAlignLeading, fyne.TextStyle{Bold: true}),
		m.bar, m.determinate, m.logScroll,

		container.NewHBox(m.startBtn, m.cancelBtn, settingsBtn),
	)

	// Top toolbar stays pinned outside the scroll container so the primary
	// actions remain reachable regardless of how far the user has scrolled
	// the Advanced accordion.
	topBar := container.NewHBox(
		m.topStartBtn, m.topCancelBtn, topSettingsBtn, readmeBtn, aboutBtn,
	)
	scrolled := container.NewScroll(container.NewPadded(form))
	w.SetContent(container.NewBorder(
		container.NewPadded(topBar), // top
		nil, nil, nil,
		scrolled, // center
	))
	w.SetCloseIntercept(m.onWindowClose)

	// Trigger initial model list
	if initial := preferredProvider(providerIDs); initial != "" {
		m.onProviderChanged(initial)
	}
	return m
}

// hasAnyAPIKey reports whether any provider has a non-blank API key in the
// loaded config. Used at launch to decide whether to auto-open Settings.
func hasAnyAPIKey(cfg ports.Config) bool {
	for _, v := range cfg.APIKeys {
		if strings.TrimSpace(v) != "" {
			return true
		}
	}
	return false
}

// preferredProvider picks ElevenLabs when configured (best parity with the
// Python CLI's default), falling back to the first available provider.
func preferredProvider(ids []string) string {
	if len(ids) == 0 {
		return ""
	}
	for _, id := range ids {
		if id == string(domain.ProviderElevenLabs) {
			return id
		}
	}
	return ids[0]
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
	svc := m.deps.Service()
	models, err := svc.ListModels(domain.ProviderID(id))
	if err != nil {
		m.logf("provider %s: %v", id, err)
		return
	}
	m.model.Options = models
	if sel := pickDefaultModel(svc.DefaultModel(domain.ProviderID(id)), models); sel != "" {
		m.model.SetSelected(sel)
	}
	m.model.Refresh()
}

// pickDefaultModel returns the provider's DefaultModel() when it appears in
// the dropdown list (always true for the hardcoded fallback; may be false
// after DiscoverModels populates a curated remote list). Falls back to the
// first list entry so the dropdown is never blank.
func pickDefaultModel(preferred string, models []string) string {
	if preferred != "" {
		for _, m := range models {
			if m == preferred {
				return preferred
			}
		}
	}
	if len(models) > 0 {
		return models[0]
	}
	return ""
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
	m.resetLog()
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

	chars := atoiOr(m.charsPerLine.Text, 0)
	words := atoiOr(m.wordsPerSubtitle.Text, 0)
	if chars > 0 && words > 0 {
		m.finishBatch(fmt.Errorf("chars-per-line and words-per-subtitle are mutually exclusive — clear one"))
		return
	}
	useCache := true
	if m.force.Checked {
		useCache = false
	}
	diarize := m.diarize.Checked
	speakerLabels := m.speakerLabels.Checked
	if !speakerLabels && diarize {
		// Mirror CLI: --speaker-labels defaults to --diarize when unset.
		speakerLabels = true
	}
	sizeMB := atofOr(m.sizeThresholdMB.Text, 100)
	req := domain.Request{
		InputPath:             file,
		Provider:              domain.ProviderID(m.provider.Selected),
		Model:                 m.model.Selected,
		Language:              strings.TrimSpace(m.language.Text),
		Formats:               formats,
		OutputDir:             strings.TrimSpace(m.outputDir.Text),
		UseCache:              useCache,
		MaxCharsPerLine:       chars,
		WordsPerSubtitle:      words,
		StartHour:             atoiOr(m.startHour.Text, 0),
		SpeakerLabels:         speakerLabels,
		NumSpeakers:           atoiOr(m.numSpeakers.Text, 0),
		KeyTerms:              csv(m.keyTermsPrompt.Text),
		SpeechModels:          csv(m.speechModels.Text),
		SizeThresholdBytes:    int64(sizeMB * 1024 * 1024),
		ChunkLengthSec:        atoiOr(m.chunkLengthSec.Text, 0),
		OverlapSec:            atoiOr(m.overlapSec.Text, 0),
		UseInput:              m.useInput.Checked,
		UsePCM:                m.usePCM.Checked,
		KeepIntermediates:     m.keep.Checked,
		KeepFLACIntermediates: m.keepFLAC.Checked,
		SaveCleanedJSON:       m.saveCleanedJSON.Checked,
	}
	if m.fmtDavinci.Checked {
		req.DaVinciOpts = &domain.DaVinciOptions{
			SilentPortionThreshold: time.Duration(atoiOr(m.silentMs.Text, 1500)) * time.Millisecond,
			PaddingStart:           time.Duration(atoiOr(m.paddingStartMs.Text, 0)) * time.Millisecond,
			PaddingEnd:             time.Duration(atoiOr(m.paddingEndMs.Text, 0)) * time.Millisecond,
			RemoveFillers:          m.removeFillers.Checked,
			SuppressFillerLines:    !m.fillerLines.Checked,
			SuppressPauses:         !m.showPauses.Checked,
			FPS:                    atofOr(m.fps.Text, 0),
			FPSOffsetStart:         atoiOr(m.fpsOffsetStart.Text, -1),
			FPSOffsetEnd:           atoiOr(m.fpsOffsetEnd.Text, 0),
			FillerWords:            csv(m.fillerWords.Text),
		}
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
	if m.fmtWordSRT.Checked {
		out = append(out, domain.FormatWordSRT)
	}
	if m.fmtDavinci.Checked {
		out = append(out, domain.FormatDavinciSRT)
	}
	return out
}

// numEntry constructs a single-line text entry suitable for numeric input.
// We don't use Fyne's validator here because empty == "use default" is the
// most common interaction; runtime parsers fall back to defaults on blank.
func numEntry(placeholder string) *widget.Entry {
	e := widget.NewEntry()
	e.SetPlaceHolder(placeholder)
	return e
}

// atoiOr parses s as an int, returning fallback when s is empty or invalid.
func atoiOr(s string, fallback int) int {
	s = strings.TrimSpace(s)
	if s == "" {
		return fallback
	}
	var n int
	if _, err := fmt.Sscanf(s, "%d", &n); err != nil {
		return fallback
	}
	return n
}

// atofOr parses s as a float64, returning fallback when s is empty or invalid.
func atofOr(s string, fallback float64) float64 {
	s = strings.TrimSpace(s)
	if s == "" {
		return fallback
	}
	var f float64
	if _, err := fmt.Sscanf(s, "%g", &f); err != nil {
		return fallback
	}
	return f
}

// csv splits a comma-separated entry into trimmed, non-empty tokens.
func csv(s string) []string {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if t := strings.TrimSpace(p); t != "" {
			out = append(out, t)
		}
	}
	if len(out) == 0 {
		return nil
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
			if sel := pickDefaultModel(svc.DefaultModel(pid), models); sel != "" {
				m.model.SetSelected(sel)
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
	set(!lock, m.startBtn, m.topStartBtn, m.provider, m.model,
		m.fmtText, m.fmtSRT, m.fmtWordSRT, m.fmtDavinci,
		m.diarize, m.speakerLabels, m.showPauses,
		m.removeFillers, m.fillerLines,
		m.useInput, m.usePCM, m.keep, m.keepFLAC,
		m.force, m.saveCleanedJSON,
	)
	set(lock, m.cancelBtn, m.topCancelBtn)
	entries := []*widget.Entry{
		m.pathEntry, m.language, m.outputDir,
		m.charsPerLine, m.wordsPerSubtitle, m.startHour, m.numSpeakers,
		m.paddingStartMs, m.paddingEndMs, m.silentMs, m.fps, m.fpsOffsetStart, m.fpsOffsetEnd,
		m.fillerWords, m.sizeThresholdMB, m.chunkLengthSec, m.overlapSec,
		m.extensions, m.keyTermsPrompt, m.speechModels,
	}
	for _, e := range entries {
		if lock {
			e.Disable()
		} else {
			e.Enable()
		}
	}
}

func (m *mainWindow) logf(format string, args ...any) {
	if m.logBuf != "" {
		m.logBuf += "\n"
	}
	m.logBuf += fmt.Sprintf(format, args...)
	m.setLogText(m.logBuf)
}

// setLogText replaces the log contents with s, rendered at full foreground
// contrast, and keeps the newest line in view.
func (m *mainWindow) setLogText(s string) {
	m.logRich.Segments = []widget.RichTextSegment{&widget.TextSegment{Text: s}}
	m.logRich.Refresh()
	m.logScroll.ScrollToBottom()
}

// resetLog clears the activity log between runs.
func (m *mainWindow) resetLog() {
	m.logBuf = ""
	m.setLogText("")
}
