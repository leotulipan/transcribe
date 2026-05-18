package tui

import (
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

func TestApp_AcceptsWindowSize(t *testing.T) {
	a := NewApp(Deps{}, Prefill{
		InputPath: "x.mp3",
		Provider:  "groq",
		Formats:   []domain.OutputFormat{domain.FormatText},
	})
	_, _ = a.Update(tea.WindowSizeMsg{Width: 80, Height: 24})
	require.Equal(t, 80, a.width)
}
