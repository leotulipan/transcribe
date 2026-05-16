package tui

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/bubbles/list"
	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

type optionsStep int

const (
	stepProvider optionsStep = iota
	stepModel
	stepLanguage
	stepFormats
)

type optionsScreen struct {
	deps Deps
	pre  Prefill
	step optionsStep
	list list.Model
	lang textinput.Model
	fmts map[domain.OutputFormat]bool
	err  error
}

func newOptions(d Deps, p Prefill) *optionsScreen {
	o := &optionsScreen{deps: d, pre: p, fmts: map[domain.OutputFormat]bool{}}
	for _, f := range p.Formats {
		o.fmts[f] = true
	}
	if p.Provider == "" {
		o.step = stepProvider
		o.list = buildProviderList(d)
	} else if p.Model == "" {
		o.step = stepModel
		o.list = buildModelList(d, p.Provider)
	} else if p.Language == "" {
		o.step = stepLanguage
		o.lang = newLanguageInput(p.Language)
	} else {
		o.step = stepFormats
		o.list = buildFormatList(o.fmts)
	}
	return o
}

func (o *optionsScreen) Init() tea.Cmd {
	if o.step == stepLanguage {
		return textinput.Blink
	}
	return nil
}

func (o *optionsScreen) Update(msg tea.Msg) (screen, tea.Cmd) {
	var cmd tea.Cmd
	if o.step == stepLanguage {
		o.lang, cmd = o.lang.Update(msg)
	} else {
		o.list, cmd = o.list.Update(msg)
	}

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
				o.step = stepLanguage
				o.lang = newLanguageInput(o.pre.Language)
				return o, textinput.Blink
			}
		case stepLanguage:
			o.pre.Language = strings.TrimSpace(o.lang.Value())
			o.step = stepFormats
			o.list = buildFormatList(o.fmts)
		case stepFormats:
			// enter doesn't advance — use space to toggle, g to submit
		}
	}
	if km, ok := msg.(tea.KeyMsg); ok && o.step == stepFormats {
		switch km.String() {
		case " ":
			if item := o.list.SelectedItem(); item != nil {
				id := domain.OutputFormat(item.(simpleItem).id)
				o.fmts[id] = !o.fmts[id]
				// Rebuild the list so the checkbox labels update
				o.list = buildFormatList(o.fmts)
			}
		case "g":
			var out []domain.OutputFormat
			for f, on := range o.fmts {
				if on {
					out = append(out, f)
				}
			}
			if len(out) == 0 {
				o.err = fmt.Errorf("pick at least one output format")
				return o, nil
			}
			o.pre.Formats = out
			return o, func() tea.Msg { return advanceMsg{pre: o.pre} }
		}
	}
	return o, cmd
}

func (o *optionsScreen) View() string {
	var head string
	switch o.step {
	case stepProvider:
		head = "Pick a provider"
	case stepModel:
		head = "Pick a model"
	case stepLanguage:
		head = "Language hint (ISO-639-1, blank = auto)"
	case stepFormats:
		head = "Output formats (space=toggle, g=go)"
	}
	body := o.list.View()
	if o.step == stepLanguage {
		body = o.lang.View()
	}
	var errLine string
	if o.err != nil {
		errLine = errorStyle.Render(o.err.Error()) + "\n"
	}
	return titleStyle.Render(head) + "\n\n" + body + "\n" + errLine
}

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

func buildFormatList(selected map[domain.OutputFormat]bool) list.Model {
	items := []list.Item{
		simpleItem{id: string(domain.FormatText), label: formatLabel(domain.FormatText, selected[domain.FormatText])},
		simpleItem{id: string(domain.FormatSRT), label: formatLabel(domain.FormatSRT, selected[domain.FormatSRT])},
		simpleItem{id: string(domain.FormatDavinciSRT), label: formatLabel(domain.FormatDavinciSRT, selected[domain.FormatDavinciSRT])},
	}
	l := list.New(items, list.NewDefaultDelegate(), 40, 8)
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

func newLanguageInput(initial string) textinput.Model {
	ti := textinput.New()
	ti.Placeholder = "en"
	ti.SetValue(initial)
	ti.Focus()
	return ti
}
