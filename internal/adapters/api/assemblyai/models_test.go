package assemblyai

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

func TestDefaultModel(t *testing.T) {
	// The default is the "auto" sentinel: it makes the client omit
	// speech_models so AssemblyAI picks its current flagship model.
	require.Equal(t, AutoModel, DefaultModel())
	require.Equal(t, "auto", AutoModel)
}

func TestModels_PriorityOrder(t *testing.T) {
	// First entry must be the default so UIs that pick models[0] still get
	// the recommended selection.
	got := Models()
	require.NotEmpty(t, got)
	require.Equal(t, DefaultModel(), got[0])
	// Besides the sentinel, only the values AssemblyAI's speech_models array
	// accepts may be advertised; everything else 400s. See models.go.
	require.Equal(t, []string{"auto", "universal-3-5-pro", "universal-2"}, got)
	// Retired on 2026-09-02 — must not be offered any more.
	require.NotContains(t, got, "universal-3-pro")
	require.NotContains(t, got, "universal-3")
	require.NotContains(t, got, "slam-1")
	require.NotContains(t, got, "best")
	require.NotContains(t, got, "nano")
}

func TestCapabilities_AutoHasWordTimestamps(t *testing.T) {
	// The GUI greys out the SRT checkboxes when the selected model reports no
	// word timestamps. "auto" is the default selection, so it must report the
	// full capability set or the default UI state would be wrong.
	c := Capabilities(AutoModel)
	require.True(t, c.WordTimestamps)
	require.False(t, c.SegmentTimestamps)
	require.True(t, c.Diarization)
	require.True(t, c.LanguageHint)
	require.Contains(t, c.AcceptedInputs, domain.AudioFormat{Codec: "mp3"})
	require.Contains(t, c.AcceptedInputs, domain.AudioFormat{Codec: "flac"})
}

func TestCapabilities_Universal35ProHasWordTimestamps(t *testing.T) {
	c := Capabilities("universal-3-5-pro")
	require.True(t, c.WordTimestamps)
	require.True(t, c.Diarization)
	require.True(t, c.LanguageHint)
	require.Contains(t, c.AcceptedInputs, domain.AudioFormat{Codec: "mp3"})
}

func TestCapabilities_Universal2Works(t *testing.T) {
	c := Capabilities("universal-2")
	require.True(t, c.WordTimestamps)
	require.True(t, c.LanguageHint)
}

func TestCapabilities_RetiredModelReturnsZero(t *testing.T) {
	// universal-3-pro is retired, and "best"/"nano" were never accepted by the
	// plural array. All are treated as unknown here; the request path remaps
	// universal-3-pro before it ever reaches Capabilities.
	for _, m := range []string{"universal-3-pro", "best", "nano"} {
		c := Capabilities(m)
		require.False(t, c.WordTimestamps, "model %q must be unknown", m)
		require.Empty(t, c.AcceptedInputs, "model %q must be unknown", m)
	}
}

func TestCapabilities_UnknownModelReturnsZero(t *testing.T) {
	c := Capabilities("totally-made-up")
	require.False(t, c.WordTimestamps)
	require.Empty(t, c.AcceptedInputs)
}

func TestNormalizeModels_RemapsDeprecated(t *testing.T) {
	out, remapped := NormalizeModels([]string{"universal-3-pro"})
	require.Equal(t, []string{"universal-3-5-pro"}, out)
	require.Equal(t, []string{"universal-3-pro"}, remapped)
}

func TestNormalizeModels_PreservesOrderAndUnknowns(t *testing.T) {
	out, remapped := NormalizeModels([]string{"universal-2", "universal-3-pro", "made-up"})
	require.Equal(t, []string{"universal-2", "universal-3-5-pro", "made-up"}, out)
	require.Equal(t, []string{"universal-3-pro"}, remapped)
}

func TestNormalizeModels_NothingToDo(t *testing.T) {
	out, remapped := NormalizeModels([]string{"universal-3-5-pro", "universal-2"})
	require.Equal(t, []string{"universal-3-5-pro", "universal-2"}, out)
	require.Nil(t, remapped)
}

func TestNormalizeModels_Idempotent(t *testing.T) {
	once, _ := NormalizeModels([]string{"universal-3-pro", "universal-2"})
	twice, remapped := NormalizeModels(once)
	require.Equal(t, once, twice)
	require.Nil(t, remapped, "second pass must report no remapping")
}

func TestNormalizeModels_Empty(t *testing.T) {
	out, remapped := NormalizeModels(nil)
	require.Nil(t, out)
	require.Nil(t, remapped)

	out, remapped = NormalizeModels([]string{})
	require.Empty(t, out)
	require.Nil(t, remapped)
}

func TestNormalizeModels_DoesNotMutateInput(t *testing.T) {
	in := []string{"universal-3-pro"}
	_, _ = NormalizeModels(in)
	require.Equal(t, []string{"universal-3-pro"}, in, "input slice must not be mutated")
}
