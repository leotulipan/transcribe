package groq

import (
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

func TestParse_Fixture(t *testing.T) {
	data, err := os.ReadFile("../../../../testdata/groq_sample.json")
	require.NoError(t, err)

	r, err := parse(data, "whisper-large-v3")
	require.NoError(t, err)

	require.Equal(t, "Hello world this is a test.", r.Text)
	require.Equal(t, "en", r.Language)
	require.Equal(t, 5*time.Second, r.Duration)
	require.Len(t, r.Words, 6)
	require.Equal(t, "Hello", r.Words[0].Text)
	require.Equal(t, 500*time.Millisecond, r.Words[0].End)
	require.Len(t, r.Segments, 2)
	require.Equal(t, domain.ProviderGroq, r.Provider)
	require.Equal(t, "whisper-large-v3", r.Model)
	require.NotEmpty(t, r.RawJSON)
}
