package elevenlabs

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

func TestDefaultModel(t *testing.T) {
	require.Equal(t, "scribe_v2", DefaultModel())
}

func TestCapabilities_DefaultModelHasWordTimestamps(t *testing.T) {
	c := Capabilities("scribe_v2")
	require.True(t, c.WordTimestamps)
	require.False(t, c.SegmentTimestamps)
	require.True(t, c.LanguageHint)
	require.Contains(t, c.AcceptedInputs, domain.AudioFormat{Codec: "mp3"})
	require.Contains(t, c.AcceptedInputs, domain.AudioFormat{Codec: "flac"})
}

func TestCapabilities_UnknownModelReturnsZero(t *testing.T) {
	c := Capabilities("totally-made-up")
	require.False(t, c.WordTimestamps)
	require.Empty(t, c.AcceptedInputs)
}
