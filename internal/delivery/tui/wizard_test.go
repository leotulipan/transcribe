package tui

import (
	"context"
	"errors"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

// fakeProvider is a test double for ports.Provider whose CheckKey is configurable.
type fakeProvider struct {
	id      domain.ProviderID
	checkFn func(ctx context.Context) error
}

func (f *fakeProvider) ID() domain.ProviderID { return f.id }
func (f *fakeProvider) MaxUploadBytes() int64  { return 0 }
func (f *fakeProvider) Models() []string       { return nil }
func (f *fakeProvider) DefaultModel() string   { return "" }
func (f *fakeProvider) Capabilities(_ string) ports.ModelCapabilities {
	return ports.ModelCapabilities{}
}
func (f *fakeProvider) Transcribe(_ context.Context, _ domain.AudioFile, _ ports.ProviderOpts) (*domain.Result, error) {
	return nil, errors.ErrUnsupported
}
func (f *fakeProvider) CheckKey(ctx context.Context) error {
	if f.checkFn != nil {
		return f.checkFn(ctx)
	}
	return nil
}

// installFakeFactory replaces providerFactory with one that returns a fake provider
// whose CheckKey uses checkFn. Restores the original on test cleanup.
func installFakeFactory(t *testing.T, checkFn func(ctx context.Context) error) {
	t.Helper()
	orig := providerFactory
	providerFactory = func(id domain.ProviderID, _ string) (ports.Provider, error) {
		return &fakeProvider{id: id, checkFn: checkFn}, nil
	}
	t.Cleanup(func() { providerFactory = orig })
}

// sendKey sends a key message to a wizardScreen and returns the updated screen.
func sendKey(w *wizardScreen, k tea.KeyType) (*wizardScreen, tea.Cmd) {
	next, cmd := w.Update(tea.KeyMsg{Type: k})
	return next.(*wizardScreen), cmd
}

// sendRune sends a rune key message to a wizardScreen.
func sendRune(w *wizardScreen, r rune) (*wizardScreen, tea.Cmd) {
	next, cmd := w.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{r}})
	return next.(*wizardScreen), cmd
}

func TestWizard_InitialStateShowsFirstProvider(t *testing.T) {
	w := newWizardScreen(ports.Config{})
	view := w.View()
	require.Contains(t, view, "groq", "first provider should be groq")
	assert.Equal(t, stateAskKey, w.state)
	assert.Equal(t, 0, w.idx)
}

func TestWizard_WalksAllSixProvidersInOrder(t *testing.T) {
	require.Equal(t, []domain.ProviderID{
		domain.ProviderGroq,
		domain.ProviderOpenAI,
		domain.ProviderAssemblyAI,
		domain.ProviderElevenLabs,
		domain.ProviderGemini,
		domain.ProviderMistral,
	}, wizardOrder)
}

func TestWizard_SkipKeyAdvances(t *testing.T) {
	w := newWizardScreen(ports.Config{})
	require.Equal(t, domain.ProviderGroq, w.currentProvider())

	w, _ = sendKey(w, tea.KeyTab) // skip groq

	assert.Equal(t, 1, w.idx)
	assert.Equal(t, domain.ProviderOpenAI, w.currentProvider())
	assert.Equal(t, stateAskKey, w.state)

	r := w.results[domain.ProviderGroq]
	assert.False(t, r.ok)
	assert.Equal(t, "[skip]", r.msg)
}

func TestWizard_EmptyEnterSkips(t *testing.T) {
	w := newWizardScreen(ports.Config{})
	// No text entered; press Enter with empty field.
	w, _ = sendKey(w, tea.KeyEnter)

	assert.Equal(t, 1, w.idx, "empty enter should skip")
	assert.Equal(t, stateAskKey, w.state)
}

func TestWizard_AdvancesAfterSuccessfulKey(t *testing.T) {
	installFakeFactory(t, nil) // CheckKey always returns nil (ok)

	w := newWizardScreen(ports.Config{})
	// Type a key value into the input.
	for _, ch := range "sk_test_key" {
		w, _ = sendRune(w, ch)
	}
	w, cmd := sendKey(w, tea.KeyEnter)
	assert.Equal(t, stateTestKey, w.state)
	require.NotNil(t, cmd)

	// Execute the command (returns checkDoneMsg).
	doneMsg := cmd().(checkDoneMsg)
	assert.True(t, doneMsg.ok)
	assert.Equal(t, domain.ProviderGroq, doneMsg.provider)

	// Feed the done message back.
	next, _ := w.Update(doneMsg)
	w = next.(*wizardScreen)

	assert.Equal(t, 1, w.idx, "should advance to next provider")
	assert.Equal(t, stateAskKey, w.state)
	assert.Equal(t, "sk_test_key", w.cfg.APIKeys[domain.ProviderGroq])
}

func TestWizard_FailedCheckDoesNotSetKey(t *testing.T) {
	installFakeFactory(t, func(_ context.Context) error {
		return errors.New("unauthorized")
	})

	w := newWizardScreen(ports.Config{})
	for _, ch := range "bad_key" {
		w, _ = sendRune(w, ch)
	}
	w, cmd := sendKey(w, tea.KeyEnter)
	doneMsg := cmd().(checkDoneMsg)
	assert.False(t, doneMsg.ok)

	next, _ := w.Update(doneMsg)
	w = next.(*wizardScreen)

	assert.Equal(t, 1, w.idx)
	assert.Equal(t, "", w.cfg.APIKeys[domain.ProviderGroq], "failed key must not be saved")
	r := w.results[domain.ProviderGroq]
	assert.True(t, strings.HasPrefix(r.msg, "[fail]"))
}

func TestWizard_UnsupportedCheckSavesKey(t *testing.T) {
	installFakeFactory(t, func(_ context.Context) error {
		return errors.ErrUnsupported
	})

	w := newWizardScreen(ports.Config{})
	for _, ch := range "my_key" {
		w, _ = sendRune(w, ch)
	}
	w, cmd := sendKey(w, tea.KeyEnter)
	doneMsg := cmd().(checkDoneMsg)
	assert.True(t, doneMsg.ok, "ErrUnsupported should be treated as ok")

	next, _ := w.Update(doneMsg)
	w = next.(*wizardScreen)

	assert.Equal(t, "my_key", w.cfg.APIKeys[domain.ProviderGroq])
}

func TestWizard_AllProvidersDoneShowsSaveScreen(t *testing.T) {
	w := newWizardScreen(ports.Config{})
	// Skip all 6 providers via Tab.
	for i := 0; i < len(wizardOrder); i++ {
		require.Equal(t, stateAskKey, w.state, "should be in stateAskKey before skip %d", i)
		w, _ = sendKey(w, tea.KeyTab)
	}
	assert.Equal(t, stateSave, w.state)
	assert.Equal(t, len(wizardOrder), w.idx)
	assert.Contains(t, w.View(), "ready to save")
}

func TestWizard_SaveConfirmEmitsDoneMsg(t *testing.T) {
	w := newWizardScreen(ports.Config{})
	// Skip all providers to reach stateSave.
	for range wizardOrder {
		w, _ = sendKey(w, tea.KeyTab)
	}
	require.Equal(t, stateSave, w.state)

	_, cmd := sendKey(w, tea.KeyEnter)
	require.NotNil(t, cmd)
	msg := cmd()
	done, ok := msg.(wizardDoneMsg)
	require.True(t, ok)
	assert.False(t, done.cancelled)
}

func TestWizard_SaveCancelEmitsCancelledMsg(t *testing.T) {
	w := newWizardScreen(ports.Config{})
	for range wizardOrder {
		w, _ = sendKey(w, tea.KeyTab)
	}
	require.Equal(t, stateSave, w.state)

	_, cmd := sendKey(w, tea.KeyEsc)
	require.NotNil(t, cmd)
	done := cmd().(wizardDoneMsg)
	assert.True(t, done.cancelled)
}

func TestWizard_ReturnsUpdatedConfigForTwoKeys(t *testing.T) {
	installFakeFactory(t, nil)

	w := newWizardScreen(ports.Config{})

	// Provider 0 (groq): enter a key and confirm.
	for _, ch := range "groq_key" {
		w, _ = sendRune(w, ch)
	}
	w, cmd := sendKey(w, tea.KeyEnter)
	done := cmd().(checkDoneMsg)
	next, _ := w.Update(done)
	w = next.(*wizardScreen)

	// Provider 1 (openai): enter a key and confirm.
	for _, ch := range "openai_key" {
		w, _ = sendRune(w, ch)
	}
	w, cmd = sendKey(w, tea.KeyEnter)
	done = cmd().(checkDoneMsg)
	next, _ = w.Update(done)
	w = next.(*wizardScreen)

	// Remaining providers: skip.
	for i := 2; i < len(wizardOrder); i++ {
		w, _ = sendKey(w, tea.KeyTab)
	}
	require.Equal(t, stateSave, w.state)

	assert.Equal(t, "groq_key", w.cfg.APIKeys[domain.ProviderGroq])
	assert.Equal(t, "openai_key", w.cfg.APIKeys[domain.ProviderOpenAI])
	assert.Equal(t, "", w.cfg.APIKeys[domain.ProviderAssemblyAI])
}

func TestWizard_ExistingKeyPrefilledInInput(t *testing.T) {
	cfg := ports.Config{
		APIKeys: map[domain.ProviderID]string{
			domain.ProviderGroq: "existing_groq",
		},
	}
	w := newWizardScreen(cfg)
	// The input should be pre-populated with the existing key.
	assert.Equal(t, "existing_groq", w.input.Value())
}
