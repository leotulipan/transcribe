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
