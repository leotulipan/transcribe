package openai

import (
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

func TestParse_Fixture(t *testing.T) {
	data, err := os.ReadFile("../../../../testdata/openai_sample.json")
	require.NoError(t, err)

	r, err := parse(data, "whisper-1")
	require.NoError(t, err)

	require.Equal(t, "Goodbye cruel world this is OpenAI.", r.Text)
	require.Equal(t, "en", r.Language)
	require.Equal(t, 4*time.Second, r.Duration)
	require.Len(t, r.Words, 6)
	require.Equal(t, "Goodbye", r.Words[0].Text)
	require.Equal(t, 600*time.Millisecond, r.Words[0].End)
	require.Len(t, r.Segments, 2)
	require.Equal(t, domain.ProviderOpenAI, r.Provider)
	require.Equal(t, "whisper-1", r.Model)
	require.NotEmpty(t, r.RawJSON)
}

func TestParse_BadJSON(t *testing.T) {
	_, err := parse([]byte("{bad json}"), "whisper-1")
	require.Error(t, err)
}
