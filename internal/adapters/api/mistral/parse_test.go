package mistral

import (
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

func TestParse_Fixture(t *testing.T) {
	data, err := os.ReadFile("../../../../testdata/mistral_sample.json")
	require.NoError(t, err)

	r, err := parse(data, "voxtral-mini-latest")
	require.NoError(t, err)

	require.Equal(t, "Bonjour monde this is Mistral.", r.Text)
	require.Equal(t, "en", r.Language)
	require.Equal(t, 3500*time.Millisecond, r.Duration)
	require.Len(t, r.Segments, 2)
	require.Equal(t, "Bonjour monde", r.Segments[0].Text)
	require.Equal(t, 1800*time.Millisecond, r.Segments[0].End)
	require.Empty(t, r.Words)
	require.Equal(t, domain.ProviderMistral, r.Provider)
	require.Equal(t, "voxtral-mini-latest", r.Model)
	require.NotEmpty(t, r.RawJSON)
}

func TestParse_BadJSON(t *testing.T) {
	_, err := parse([]byte("{bad json}"), "voxtral-mini-latest")
	require.Error(t, err)
}
