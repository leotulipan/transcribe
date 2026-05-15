package elevenlabs

import (
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

func TestParse_Fixture(t *testing.T) {
	data, err := os.ReadFile("../../../../testdata/elevenlabs_sample.json")
	require.NoError(t, err)

	r, err := parse(data, "scribe_v1")
	require.NoError(t, err)

	require.Equal(t, "Hello world", r.Text)
	require.Equal(t, "eng", r.Language)
	// Words: only "word" type entries (spacing filtered)
	require.Len(t, r.Words, 2)
	require.Equal(t, "Hello", r.Words[0].Text)
	require.Equal(t, 100*time.Millisecond, r.Words[0].Start)
	require.Equal(t, 600*time.Millisecond, r.Words[0].End)
	require.Equal(t, "world", r.Words[1].Text)
	require.Equal(t, 700*time.Millisecond, r.Words[1].Start)
	require.Equal(t, 1200*time.Millisecond, r.Words[1].End)
	require.Equal(t, domain.ProviderElevenLabs, r.Provider)
	require.Equal(t, "scribe_v1", r.Model)
	require.NotEmpty(t, r.RawJSON)
}

func TestParse_BadJSON(t *testing.T) {
	_, err := parse([]byte("{bad json}"), "scribe_v1")
	require.Error(t, err)
}
