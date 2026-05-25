package tui

import (
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

// sendOptionsKey sends a key to an optionsScreen and returns the updated screen.
func sendOptionsKey(o *optionsScreen, k tea.KeyType) (*optionsScreen, tea.Cmd) {
	next, cmd := o.Update(tea.KeyMsg{Type: k})
	return next.(*optionsScreen), cmd
}

func sendOptionsRune(o *optionsScreen, r rune) (*optionsScreen, tea.Cmd) {
	next, cmd := o.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{r}})
	return next.(*optionsScreen), cmd
}

func TestOptions_FormatsIncludeWordSRT(t *testing.T) {
	// Start at formats step directly.
	o := &optionsScreen{
		step: stepFormats,
		fmts: map[domain.OutputFormat]bool{},
	}
	o.list = buildFormatList(o.fmts)

	// Collect all item IDs from the list.
	var ids []string
	for _, item := range o.list.Items() {
		ids = append(ids, item.(simpleItem).id)
	}

	require.Contains(t, ids, string(domain.FormatWordSRT), "word_srt must appear in the format list")
	require.Contains(t, ids, string(domain.FormatText))
	require.Contains(t, ids, string(domain.FormatSRT))
	require.Contains(t, ids, string(domain.FormatDavinciSRT))
}

func TestOptions_FormatsOrder(t *testing.T) {
	o := &optionsScreen{
		step: stepFormats,
		fmts: map[domain.OutputFormat]bool{},
	}
	o.list = buildFormatList(o.fmts)
	items := o.list.Items()
	require.Len(t, items, 4)
	assert.Equal(t, string(domain.FormatText), items[0].(simpleItem).id)
	assert.Equal(t, string(domain.FormatSRT), items[1].(simpleItem).id)
	assert.Equal(t, string(domain.FormatWordSRT), items[2].(simpleItem).id)
	assert.Equal(t, string(domain.FormatDavinciSRT), items[3].(simpleItem).id)
}

func TestOptions_LanguagePickerEnumeratesCommonCodes(t *testing.T) {
	o := &optionsScreen{
		step: stepLanguage,
		fmts: map[domain.OutputFormat]bool{},
	}
	o.list = buildLanguageList()

	items := o.list.Items()
	require.True(t, len(items) >= 16, "must have auto + 15 ISO codes")

	var ids []string
	for _, it := range items {
		ids = append(ids, it.(simpleItem).id)
	}

	expected := []string{"", "en", "de", "es", "fr", "it", "pt", "nl", "pl", "sv", "no", "da", "fi", "ja", "ko", "zh"}
	for _, code := range expected {
		assert.Contains(t, ids, code, "language list must include %q", code)
	}
	// "auto" must be first.
	assert.Equal(t, "", items[0].(simpleItem).id, "first item id must be empty string (auto)")
}

func TestOptions_AdvancedScreenOpens(t *testing.T) {
	o := &optionsScreen{
		step: stepFormats,
		fmts: map[domain.OutputFormat]bool{domain.FormatText: true},
	}
	o.list = buildFormatList(o.fmts)

	// Press 'a' to enter the advanced flow.
	next, _ := sendOptionsRune(o, 'a')
	assert.Equal(t, stepAdvanced, next.step, "pressing 'a' on formats must advance to stepAdvanced")
}

func TestOptions_AdvancedDefaults(t *testing.T) {
	o := &optionsScreen{
		step: stepAdvanced,
		pre:  Prefill{},
		fmts: map[domain.OutputFormat]bool{domain.FormatText: true},
	}
	o.adv = defaultAdvancedOpts()
	o.list = buildAdvancedList(o.adv)

	assert.False(t, o.adv.Diarize, "diarize must default to false")
	assert.False(t, o.adv.RemoveFillers, "remove-fillers must default to false")
	assert.True(t, o.adv.FillerLines, "filler-lines must default to true")
	assert.Equal(t, 0, o.adv.PaddingStartMs, "padding-start must default to 0")
	assert.Equal(t, 0, o.adv.PaddingEndMs, "padding-end must default to 0")
}

func TestOptions_AdvancedSubmissionBuildsRequest(t *testing.T) {
	deps := Deps{Service: &fakeSvc{}}
	pre := Prefill{
		InputPath: "clip.mp3",
		Provider:  domain.ProviderGroq,
		Model:     "whisper-large-v3",
		Language:  "en",
		Formats:   []domain.OutputFormat{domain.FormatDavinciSRT},
	}
	a := NewApp(deps, pre)

	// Should be at stepFormats since formats are provided but we set them in pre.
	// Actually with full Formats pre-filled, App skips to progress.
	// Simulate having walked to advanced step manually.
	opts := optionsScreen{
		deps: deps,
		pre:  pre,
		step: stepAdvanced,
		fmts: map[domain.OutputFormat]bool{domain.FormatDavinciSRT: true},
		adv: advancedOpts{
			Diarize:        true,
			RemoveFillers:  true,
			FillerLines:    true,
			PaddingStartMs: 100,
			PaddingEndMs:   50,
		},
	}
	opts.list = buildAdvancedList(opts.adv)

	// Press 'g' to submit.
	var advanceReceived advanceMsg
	next, cmd := sendOptionsRune(&opts, 'g')
	_ = next
	require.NotNil(t, cmd)
	msg := cmd()
	adv, ok := msg.(advanceMsg)
	require.True(t, ok, "pressing g on advanced must emit advanceMsg")
	advanceReceived = adv

	// The pre in the advanceMsg should carry the advanced flags.
	assert.True(t, advanceReceived.pre.Diarize)
	assert.True(t, advanceReceived.pre.RemoveFillers)
	assert.True(t, advanceReceived.pre.FillerLines)
	assert.Equal(t, 100, advanceReceived.pre.PaddingStartMs)
	assert.Equal(t, 50, advanceReceived.pre.PaddingEndMs)

	// Verify buildRequest picks them up.
	a.pre = advanceReceived.pre
	req := a.buildRequest()
	assert.True(t, req.SpeakerLabels, "SpeakerLabels must be set when Diarize=true")
	require.NotNil(t, req.DaVinciOpts, "DaVinciOpts must be populated for davinci_srt")
	assert.True(t, req.DaVinciOpts.RemoveFillers)
	assert.False(t, req.DaVinciOpts.SuppressFillerLines, "FillerLines=true → SuppressFillerLines=false")
}

func TestApp_PrefilledFromCLISkipsAdvanced(t *testing.T) {
	// When CLI pre-fills diarize=true, the TUI should not force the user through
	// the advanced screen — the value is carried in Prefill and buildRequest picks it up.
	pre := Prefill{
		InputPath:      "clip.mp3",
		Provider:       domain.ProviderGroq,
		Model:          "whisper-large-v3",
		Language:       "en",
		Formats:        []domain.OutputFormat{domain.FormatDavinciSRT},
		Diarize:        true,
		RemoveFillers:  false,
		FillerLines:    true,
		PaddingStartMs: 200,
		PaddingEndMs:   0,
	}
	a := NewApp(Deps{Service: &fakeSvc{}}, pre)
	// Full prefill → goes straight to progress, skipping advanced entirely.
	assert.Equal(t, screenProgress, a.curID, "full prefill must skip to progress screen")

	req := a.buildRequest()
	assert.True(t, req.SpeakerLabels)
	require.NotNil(t, req.DaVinciOpts)
	assert.Equal(t, 200, int(req.DaVinciOpts.PaddingStart.Milliseconds()))
}
