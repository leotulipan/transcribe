package gemini

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

func TestDefaultModel(t *testing.T) {
	require.Equal(t, "gemini-2.0-flash", DefaultModel())
}

func TestCapabilities_DefaultModelIsTextOnly(t *testing.T) {
	c := Capabilities("gemini-2.0-flash")
	require.False(t, c.WordTimestamps)
	require.False(t, c.SegmentTimestamps)
	require.False(t, c.LanguageHint)
	require.Contains(t, c.AcceptedInputs, domain.AudioFormat{Codec: "mp3"})
	require.Contains(t, c.AcceptedInputs, domain.AudioFormat{Codec: "flac"})
}

func TestCapabilities_UnknownModelReturnsZero(t *testing.T) {
	c := Capabilities("totally-made-up")
	require.False(t, c.WordTimestamps)
	require.Empty(t, c.AcceptedInputs)
}
