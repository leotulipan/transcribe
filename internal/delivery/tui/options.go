package tui

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/charmbracelet/bubbles/list"
	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"

	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

type optionsStep int

const (
	stepProvider optionsStep = iota
	stepModel
	stepLanguage
	stepFormats
	stepAdvanced
)

// advancedOpts holds the optional flags surfaced on the advanced screen.
type advancedOpts struct {
	Diarize        bool
	RemoveFillers  bool
	FillerLines    bool // positive form — true means emit UPPERCASE filler lines
	PaddingStartMs int
	PaddingEndMs   int
}

func defaultAdvancedOpts() advancedOpts {
	return advancedOpts{
		FillerLines: true, // mirrors CLI default --filler-lines=true
	}
}

// advancedField indexes the navigable rows on the advanced screen.
type advancedField int

const (
	advFieldDiarize advancedField = iota
	advFieldRemoveFillers
	advFieldFillerLines
	advFieldPaddingStart
	advFieldPaddingEnd
	advFieldCount // sentinel — keep last
)

type optionsScreen struct {
	deps      Deps
	pre       Prefill
	step      optionsStep
	list      list.Model
	fmts      map[domain.OutputFormat]bool
	caps      ports.ModelCapabilities // capabilities of the selected provider/model
	adv       advancedOpts
	advField  advancedField   // selected row in the advanced screen
	padInputs [2]textinput.Model // [0]=paddingStart, [1]=paddingEnd
	err       error
}

func newOptions(d Deps, p Prefill) *optionsScreen {
	o := &optionsScreen{deps: d, pre: p, fmts: map[domain.OutputFormat]bool{}}
	for _, f := range p.Formats {
		o.fmts[f] = true
	}
	// Seed advanced opts from prefill so CLI flags survive the options flow.
	o.adv = advancedOpts{
		Diarize:        p.Diarize,
		RemoveFillers:  p.RemoveFillers,
		FillerLines:    p.FillerLines,
		PaddingStartMs: p.PaddingStartMs,
		PaddingEndMs:   p.PaddingEndMs,
	}
	if p.Provider == "" {
		o.step = stepProvider
		o.list = buildProviderList(d)
	} else if p.Model == "" {
		o.step = stepModel
		o.list = buildModelList(d, p.Provider)
	} else if p.Language == "" {
		o.step = stepLanguage
		o.list = buildLanguageList()
	} else {
		o.step = stepFormats
		o.refreshCaps()
		o.applyFormatCaps()
		o.list = buildFormatList(o.fmts, o.caps)
	}
	return o
}

// refreshCaps loads the capabilities of the currently selected provider/model.
// Defaults to permissive (everything shown) when they can't be determined, so
// the server-side check — not the UI — is the final gate.
func (o *optionsScreen) refreshCaps() {
	o.caps = ports.ModelCapabilities{WordTimestamps: true, Diarization: true}
	if o.deps.Service != nil && o.pre.Provider != "" && o.pre.Model != "" {
		if c, ok := o.deps.Service.Capabilities(o.pre.Provider, o.pre.Model); ok {
			o.caps = c
		}
	}
}

// applyFormatCaps unchecks SRT-family formats the model can't produce, and notes
// it so the user understands why the list shrank.
func (o *optionsScreen) applyFormatCaps() {
	if o.caps.WordTimestamps {
		return
	}
	dropped := false
	for _, f := range []domain.OutputFormat{domain.FormatSRT, domain.FormatWordSRT, domain.FormatDavinciSRT} {
		if o.fmts[f] {
			o.fmts[f] = false
			dropped = true
		}
	}
	if dropped {
		o.err = fmt.Errorf("%s outputs plain text only — SRT formats are unavailable", o.pre.Model)
	}
}

// firstAdvField is the topmost selectable row on the advanced screen. The
// diarization row is hidden for providers that don't support it.
func (o *optionsScreen) firstAdvField() advancedField {
	if !o.caps.Diarization {
		return advFieldRemoveFillers
	}
	return advFieldDiarize
}

func (o *optionsScreen) Init() tea.Cmd {
	if o.step == stepAdvanced {
		return o.padInputCmd()
	}
	return nil
}

func (o *optionsScreen) padInputCmd() tea.Cmd {
	if o.advField == advFieldPaddingStart {
		return o.padInputs[0].Focus()
	}
	if o.advField == advFieldPaddingEnd {
		return o.padInputs[1].Focus()
	}
	return nil
}

func (o *optionsScreen) Update(msg tea.Msg) (screen, tea.Cmd) {
	if o.step == stepAdvanced {
		return o.updateAdvanced(msg)
	}
	return o.updateOptions(msg)
}

func (o *optionsScreen) updateOptions(msg tea.Msg) (screen, tea.Cmd) {
	var cmd tea.Cmd
	o.list, cmd = o.list.Update(msg)

	if km, ok := msg.(tea.KeyMsg); ok && km.Type == tea.KeyEnter {
		switch o.step {
		case stepProvider:
			if item := o.list.SelectedItem(); item != nil {
				id := item.(simpleItem).id
				o.pre.Provider = domain.ProviderID(id)
				o.step = stepModel
				o.list = buildModelList(o.deps, o.pre.Provider)
			}
		case stepModel:
			if item := o.list.SelectedItem(); item != nil {
				o.pre.Model = item.(simpleItem).id
				o.refreshCaps()
				o.step = stepLanguage
				o.list = buildLanguageList()
			}
		case stepLanguage:
			if item := o.list.SelectedItem(); item != nil {
				o.pre.Language = item.(simpleItem).id // empty string = auto
				o.step = stepFormats
				o.applyFormatCaps()
				o.list = buildFormatList(o.fmts, o.caps)
			}
		case stepFormats:
			// Enter doesn't advance — use space to toggle, g to submit, a for advanced.
		}
	}
	if km, ok := msg.(tea.KeyMsg); ok && o.step == stepFormats {
		switch km.String() {
		case " ":
			if item := o.list.SelectedItem(); item != nil {
				id := domain.OutputFormat(item.(simpleItem).id)
				o.fmts[id] = !o.fmts[id]
				o.list = buildFormatList(o.fmts, o.caps)
			}
		case "a":
			// Enter the advanced options screen before submitting.
			o.adv.PaddingStartMs = o.pre.PaddingStartMs
			o.adv.PaddingEndMs = o.pre.PaddingEndMs
			o.step = stepAdvanced
			o.advField = o.firstAdvField()
			o.padInputs = buildPadInputs(o.adv)
			o.list = buildAdvancedList(o.adv)
			return o, nil
		case "g":
			return o, o.submitFormats()
		}
	}
	return o, cmd
}

// submitFormats validates the selected formats and emits advanceMsg.
func (o *optionsScreen) submitFormats() tea.Cmd {
	var out []domain.OutputFormat
	for f, on := range o.fmts {
		if on {
			out = append(out, f)
		}
	}
	if len(out) == 0 {
		o.err = fmt.Errorf("pick at least one output format")
		return nil
	}
	o.pre.Formats = out
	return func() tea.Msg { return advanceMsg{pre: o.pre} }
}

func (o *optionsScreen) updateAdvanced(msg tea.Msg) (screen, tea.Cmd) {
	// Update active text inputs when on a padding field.
	var cmd tea.Cmd
	switch o.advField {
	case advFieldPaddingStart:
		o.padInputs[0], cmd = o.padInputs[0].Update(msg)
	case advFieldPaddingEnd:
		o.padInputs[1], cmd = o.padInputs[1].Update(msg)
	default:
		o.list, cmd = o.list.Update(msg)
	}

	km, isKey := msg.(tea.KeyMsg)
	if !isKey {
		return o, cmd
	}

	switch km.String() {
	case "esc":
		// Go back to format selection.
		o.step = stepFormats
		o.list = buildFormatList(o.fmts, o.caps)
		return o, nil
	case "g":
		// Commit padding text inputs before submitting.
		o.commitPadInputs()
		o.pre.Diarize = o.adv.Diarize
		o.pre.RemoveFillers = o.adv.RemoveFillers
		o.pre.FillerLines = o.adv.FillerLines
		o.pre.PaddingStartMs = o.adv.PaddingStartMs
		o.pre.PaddingEndMs = o.adv.PaddingEndMs
		return o, o.submitFormats()
	case " ":
		// Toggle boolean fields.
		switch o.advField {
		case advFieldDiarize:
			o.adv.Diarize = !o.adv.Diarize
		case advFieldRemoveFillers:
			o.adv.RemoveFillers = !o.adv.RemoveFillers
		case advFieldFillerLines:
			o.adv.FillerLines = !o.adv.FillerLines
		}
		o.list = buildAdvancedList(o.adv)
	case "up", "k":
		if o.advField > o.firstAdvField() {
			o.advField--
			o.padInputs = buildPadInputs(o.adv)
			o.list = buildAdvancedList(o.adv)
			return o, o.padInputCmd()
		}
	case "down", "j":
		if o.advField < advFieldCount-1 {
			o.advField++
			o.padInputs = buildPadInputs(o.adv)
			o.list = buildAdvancedList(o.adv)
			return o, o.padInputCmd()
		}
	case "enter":
		// On padding fields, commit the typed value.
		if o.advField == advFieldPaddingStart || o.advField == advFieldPaddingEnd {
			o.commitPadInputs()
			o.list = buildAdvancedList(o.adv)
		}
	}
	return o, cmd
}

// commitPadInputs parses the text-input values back into adv.
func (o *optionsScreen) commitPadInputs() {
	if v, err := strconv.Atoi(strings.TrimSpace(o.padInputs[0].Value())); err == nil {
		o.adv.PaddingStartMs = v
	}
	if v, err := strconv.Atoi(strings.TrimSpace(o.padInputs[1].Value())); err == nil {
		o.adv.PaddingEndMs = v
	}
}

func (o *optionsScreen) View() string {
	var head string
	switch o.step {
	case stepProvider:
		head = "Pick a provider"
	case stepModel:
		head = "Pick a model"
	case stepLanguage:
		head = "Pick a language"
	case stepFormats:
		head = "Output formats  (space=toggle  a=advanced  g=go)"
	case stepAdvanced:
		head = "Advanced options  (space=toggle  ↑/↓=navigate  g=submit  esc=back)"
	}

	var body string
	switch o.step {
	case stepAdvanced:
		body = o.viewAdvanced()
	default:
		body = o.list.View()
	}

	var errLine string
	if o.err != nil {
		errLine = errorStyle.Render(o.err.Error()) + "\n"
	}
	return titleStyle.Render(head) + "\n\n" + body + "\n" + errLine
}

// viewAdvanced renders the advanced options screen manually so we can mix
// list rows (booleans) with text inputs (padding) in a single view.
func (o *optionsScreen) viewAdvanced() string {
	type advRow struct {
		field advancedField
		label string
		value string
	}
	var rows []advRow
	// Diarization is only offered by providers that support it.
	if o.caps.Diarization {
		rows = append(rows, advRow{advFieldDiarize, "Speaker diarization", boolMark(o.adv.Diarize)})
	}
	rows = append(rows,
		advRow{advFieldRemoveFillers, "Remove filler words", boolMark(o.adv.RemoveFillers)},
		advRow{advFieldFillerLines, "Filler word lines (DaVinci)", boolMark(o.adv.FillerLines)},
		advRow{advFieldPaddingStart, "Padding start (ms)", o.padInputs[0].View()},
		advRow{advFieldPaddingEnd, "Padding end (ms)", o.padInputs[1].View()},
	)
	var sb strings.Builder
	for _, r := range rows {
		cursor := "  "
		if r.field == o.advField {
			cursor = helpStyle.Render("▶ ")
		}
		sb.WriteString(cursor + r.label + ": " + r.value + "\n")
	}
	return sb.String()
}

func boolMark(v bool) string {
	if v {
		return "[x]"
	}
	return "[ ]"
}

// ---- builders ---------------------------------------------------------------

type simpleItem struct{ id, label string }

func (i simpleItem) Title() string       { return i.label }
func (i simpleItem) Description() string { return "" }
func (i simpleItem) FilterValue() string { return i.label }

func buildProviderList(d Deps) list.Model {
	var items []list.Item
	if d.Service != nil {
		for _, id := range d.Service.ListProviders() {
			items = append(items, simpleItem{id: string(id), label: string(id)})
		}
	}
	l := list.New(items, list.NewDefaultDelegate(), 40, 12)
	l.Title = "Providers"
	return l
}

func buildModelList(d Deps, p domain.ProviderID) list.Model {
	var items []list.Item
	if d.Service != nil {
		models, _ := d.Service.ListModels(p)
		for _, m := range models {
			items = append(items, simpleItem{id: m, label: m})
		}
	}
	l := list.New(items, list.NewDefaultDelegate(), 40, 12)
	l.Title = "Models for " + string(p)
	return l
}

// commonLanguages is the dropdown content for the language picker.
// The empty-string ID represents "auto" (no hint).
var commonLanguages = []simpleItem{
	{id: "", label: "auto (detect)"},
	{id: "en", label: "en — English"},
	{id: "de", label: "de — German"},
	{id: "es", label: "es — Spanish"},
	{id: "fr", label: "fr — French"},
	{id: "it", label: "it — Italian"},
	{id: "pt", label: "pt — Portuguese"},
	{id: "nl", label: "nl — Dutch"},
	{id: "pl", label: "pl — Polish"},
	{id: "sv", label: "sv — Swedish"},
	{id: "no", label: "no — Norwegian"},
	{id: "da", label: "da — Danish"},
	{id: "fi", label: "fi — Finnish"},
	{id: "ja", label: "ja — Japanese"},
	{id: "ko", label: "ko — Korean"},
	{id: "zh", label: "zh — Chinese"},
}

func buildLanguageList() list.Model {
	items := make([]list.Item, len(commonLanguages))
	for i, lang := range commonLanguages {
		items[i] = lang
	}
	l := list.New(items, list.NewDefaultDelegate(), 40, 14)
	l.Title = "Language"
	return l
}

// buildFormatList builds the output-format picker. Timestamp-based formats
// (srt, word_srt, davinci_srt) are omitted when the model can't produce
// word-level timestamps — only plain text is offered then.
func buildFormatList(selected map[domain.OutputFormat]bool, caps ports.ModelCapabilities) list.Model {
	items := []list.Item{
		simpleItem{id: string(domain.FormatText), label: formatLabel(domain.FormatText, selected[domain.FormatText])},
	}
	if caps.WordTimestamps {
		items = append(items,
			simpleItem{id: string(domain.FormatSRT), label: formatLabel(domain.FormatSRT, selected[domain.FormatSRT])},
			simpleItem{id: string(domain.FormatWordSRT), label: formatLabel(domain.FormatWordSRT, selected[domain.FormatWordSRT])},
			simpleItem{id: string(domain.FormatDavinciSRT), label: formatLabel(domain.FormatDavinciSRT, selected[domain.FormatDavinciSRT])},
		)
	}
	l := list.New(items, list.NewDefaultDelegate(), 40, 10)
	l.Title = "Outputs"
	return l
}

func formatLabel(f domain.OutputFormat, on bool) string {
	mark := "[ ]"
	if on {
		mark = "[x]"
	}
	return mark + " " + string(f)
}

// buildAdvancedList builds the list.Model for the boolean toggles portion of the
// advanced screen (used for key routing only — the view is rendered manually).
func buildAdvancedList(adv advancedOpts) list.Model {
	items := []list.Item{
		simpleItem{id: "diarize", label: boolMark(adv.Diarize) + " Speaker diarization"},
		simpleItem{id: "remove_fillers", label: boolMark(adv.RemoveFillers) + " Remove filler words"},
		simpleItem{id: "filler_lines", label: boolMark(adv.FillerLines) + " Filler word lines (DaVinci)"},
		simpleItem{id: "padding_start", label: "Padding start (ms)"},
		simpleItem{id: "padding_end", label: "Padding end (ms)"},
	}
	l := list.New(items, list.NewDefaultDelegate(), 40, 12)
	l.Title = "Advanced"
	return l
}

func buildPadInputs(adv advancedOpts) [2]textinput.Model {
	start := textinput.New()
	start.Placeholder = "0"
	start.SetValue(strconv.Itoa(adv.PaddingStartMs))
	start.CharLimit = 6

	end := textinput.New()
	end.Placeholder = "0"
	end.SetValue(strconv.Itoa(adv.PaddingEndMs))
	end.CharLimit = 6

	return [2]textinput.Model{start, end}
}
