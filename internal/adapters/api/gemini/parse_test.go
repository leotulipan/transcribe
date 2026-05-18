package gemini

import (
	"os"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

func TestParse_Fixture(t *testing.T) {
	data, err := os.ReadFile("../../../../testdata/gemini_sample.json")
	require.NoError(t, err)

	r, err := parse(data, "gemini-2.0-flash")
	require.NoError(t, err)

	require.Equal(t, "Hello world", r.Text)
	require.Empty(t, r.Words)
	require.Empty(t, r.Segments)
	require.Equal(t, domain.ProviderGemini, r.Provider)
	require.Equal(t, "gemini-2.0-flash", r.Model)
	require.NotEmpty(t, r.RawJSON)
}

func TestParse_NoCandidates(t *testing.T) {
	_, err := parse([]byte(`{"candidates": []}`), "gemini-2.0-flash")
	require.Error(t, err)
	require.Contains(t, err.Error(), "no candidates")
}

func TestParse_BadJSON(t *testing.T) {
	_, err := parse([]byte("{bad json}"), "gemini-2.0-flash")
	require.Error(t, err)
}
