package assemblyai

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

func TestDefaultModel(t *testing.T) {
	require.Equal(t, "universal-3-pro", DefaultModel())
}

func TestModels_PriorityOrder(t *testing.T) {
	// First entry must be the default so UIs that pick models[0] still get
	// the recommended model.
	got := Models()
	require.NotEmpty(t, got)
	require.Equal(t, DefaultModel(), got[0])
	// Only the values AssemblyAI's speech_models array accepts may be
	// advertised; everything else 400s ("must be a non-empty list ... of:
	// universal-3-pro, universal-2"). See models.go.
	require.Equal(t, []string{"universal-3-pro", "universal-2"}, got)
	// The non-pro "universal-3" and the legacy singular speech_model values
	// are rejected by the plural array, so they must not be offered.
	require.NotContains(t, got, "universal-3")
	require.NotContains(t, got, "slam-1")
	require.NotContains(t, got, "best")
	require.NotContains(t, got, "nano")
}

func TestCapabilities_DefaultModelHasWordTimestamps(t *testing.T) {
	c := Capabilities("universal-3-pro")
	require.True(t, c.WordTimestamps)
	require.False(t, c.SegmentTimestamps)
	require.True(t, c.LanguageHint)
	require.Contains(t, c.AcceptedInputs, domain.AudioFormat{Codec: "mp3"})
	require.Contains(t, c.AcceptedInputs, domain.AudioFormat{Codec: "flac"})
}

func TestCapabilities_Universal2Works(t *testing.T) {
	c := Capabilities("universal-2")
	require.True(t, c.WordTimestamps)
	require.True(t, c.LanguageHint)
}

func TestCapabilities_RetiredModelReturnsZero(t *testing.T) {
	// "best"/"nano" are no longer accepted by the speech_models array, so they
	// are treated as unknown.
	c := Capabilities("best")
	require.False(t, c.WordTimestamps)
	require.Empty(t, c.AcceptedInputs)
}

func TestCapabilities_UnknownModelReturnsZero(t *testing.T) {
	c := Capabilities("totally-made-up")
	require.False(t, c.WordTimestamps)
	require.Empty(t, c.AcceptedInputs)
}
