package assemblyai

import (
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

func TestParse_Fixture(t *testing.T) {
	data, err := os.ReadFile("../../../../testdata/assemblyai_sample.json")
	require.NoError(t, err)

	r, err := parse(data, "best")
	require.NoError(t, err)

	require.Equal(t, "Hello world", r.Text)
	require.Equal(t, "en", r.Language)
	require.Equal(t, 5*time.Second, r.Duration)
	require.Len(t, r.Words, 2)
	require.Equal(t, 100*time.Millisecond, r.Words[0].Start)
	require.Equal(t, 600*time.Millisecond, r.Words[0].End)
	require.Equal(t, 1200*time.Millisecond, r.Words[1].End)
	require.Equal(t, domain.ProviderAssemblyAI, r.Provider)
	require.Equal(t, "best", r.Model)
	require.NotEmpty(t, r.RawJSON)
}

func TestParse_ErrorStatus(t *testing.T) {
	data := []byte(`{"id":"x","status":"error","error":"something went wrong"}`)
	_, err := parse(data, "best")
	require.Error(t, err)
	require.Contains(t, err.Error(), "something went wrong")
}

func TestParse_BadJSON(t *testing.T) {
	_, err := parse([]byte("{bad json}"), "best")
	require.Error(t, err)
}

func TestParse_AutoRecordsActualModel(t *testing.T) {
	// In auto mode we never told the API which model to use, so the result
	// must name whatever it actually ran rather than the "auto" sentinel.
	data := []byte(`{"id":"x","status":"completed","text":"hi","language_code":"en","speech_model":"universal-3-5-pro","audio_duration":1}`)
	r, err := parse(data, AutoModel)
	require.NoError(t, err)
	require.Equal(t, "universal-3-5-pro", r.Model)
}

func TestParse_AutoWithoutEchoKeepsSentinel(t *testing.T) {
	// Older responses omit speech_model; the sentinel is the honest fallback.
	data := []byte(`{"id":"x","status":"completed","text":"hi","language_code":"en","audio_duration":1}`)
	r, err := parse(data, AutoModel)
	require.NoError(t, err)
	require.Equal(t, AutoModel, r.Model)
}

func TestParse_PinnedModelIgnoresEcho(t *testing.T) {
	// When the user pinned a model, the requested value is what we report —
	// the echo can carry legacy labels like "best".
	data := []byte(`{"id":"x","status":"completed","text":"hi","language_code":"en","speech_model":"best","audio_duration":1}`)
	r, err := parse(data, "universal-2")
	require.NoError(t, err)
	require.Equal(t, "universal-2", r.Model)
}
