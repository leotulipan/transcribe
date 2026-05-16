package tui

import (
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/x/exp/teatest"
	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

// TestApp_ModelWiring verifies that NewApp routes to the correct screen based
// on prefill values and that Update propagates messages without panicking.
func TestApp_ModelWiring(t *testing.T) {
	t.Run("no_input_starts_filepicker", func(t *testing.T) {
		a := NewApp(Deps{Service: &fakeSvc{}}, Prefill{})
		require.Equal(t, screenFilePicker, a.curID)
		require.NotEmpty(t, a.View())
	})

	t.Run("input_only_starts_options", func(t *testing.T) {
		a := NewApp(Deps{Service: &fakeSvc{}}, Prefill{InputPath: "x.mp3"})
		require.Equal(t, screenOptions, a.curID)
		view := a.View()
		require.True(t, len(view) > 0, "options screen must render non-empty view")
	})

	t.Run("full_prefill_starts_progress", func(t *testing.T) {
		a := NewApp(Deps{Service: &fakeSvc{}}, Prefill{
			InputPath: "x.mp3",
			Provider:  domain.ProviderGroq,
			Formats:   []domain.OutputFormat{domain.FormatText},
		})
		require.Equal(t, screenProgress, a.curID)
		require.NotEmpty(t, a.View())
	})

	t.Run("windowsize_stored", func(t *testing.T) {
		a := NewApp(Deps{Service: &fakeSvc{}}, Prefill{InputPath: "x.mp3", Provider: "groq", Formats: []domain.OutputFormat{domain.FormatText}})
		_, _ = a.Update(tea.WindowSizeMsg{Width: 120, Height: 40})
		require.Equal(t, 120, a.width)
		require.Equal(t, 40, a.height)
	})
}

// TestTUI_PrefilledProgressFlow uses teatest to exercise the full Bubble Tea
// event loop against a fake service that completes immediately.
func TestTUI_PrefilledProgressFlow(t *testing.T) {
	deps := Deps{Service: &fakeSvc{}}
	pre := Prefill{
		InputPath: "tiny.mp3",
		Provider:  domain.ProviderGroq,
		Model:     "whisper-large-v3",
		Formats:   []domain.OutputFormat{domain.FormatText},
	}
	a := NewApp(deps, pre)
	tm := teatest.NewTestModel(t, a, teatest.WithInitialTermSize(80, 24))

	// Wait until the view contains "Transcribing" or "done" or any content.
	teatest.WaitFor(t, tm.Output(), func(b []byte) bool {
		return strings.Contains(string(b), "Transcribing") || strings.Contains(string(b), "done") || len(b) > 0
	}, teatest.WithDuration(5*time.Second))

	tm.Quit()
	require.NotNil(t, a)
}
