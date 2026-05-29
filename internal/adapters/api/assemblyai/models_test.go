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
	// Legacy IDs stay available so older configs keep working.
	require.Contains(t, got, "best")
	require.Contains(t, got, "nano")
}

func TestCapabilities_DefaultModelHasWordTimestamps(t *testing.T) {
	c := Capabilities("universal-3-pro")
	require.True(t, c.WordTimestamps)
	require.False(t, c.SegmentTimestamps)
	require.True(t, c.LanguageHint)
	require.Contains(t, c.AcceptedInputs, domain.AudioFormat{Codec: "mp3"})
	require.Contains(t, c.AcceptedInputs, domain.AudioFormat{Codec: "flac"})
}

func TestCapabilities_LegacyModelStillWorks(t *testing.T) {
	c := Capabilities("best")
	require.True(t, c.WordTimestamps)
	require.True(t, c.LanguageHint)
}

func TestCapabilities_UnknownModelReturnsZero(t *testing.T) {
	c := Capabilities("totally-made-up")
	require.False(t, c.WordTimestamps)
	require.Empty(t, c.AcceptedInputs)
}
