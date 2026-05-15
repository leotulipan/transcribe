//go:build integration

package assemblyai

import (
	"context"
	"net/http"
	"os"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

func TestIntegration_AssemblyAI_Transcribe(t *testing.T) {
	key := os.Getenv("TRANSCRIBE_ASSEMBLYAI_KEY")
	if key == "" {
		t.Skip("TRANSCRIBE_ASSEMBLYAI_KEY not set")
	}
	c := New(key, http.DefaultClient)
	res, err := c.Transcribe(context.Background(),
		domain.AudioFile{Path: "../../../../testdata/short-sample.mp3", Container: "mp3", Codec: "mp3"},
		ports.ProviderOpts{Model: c.DefaultModel(), Language: "en"},
	)
	require.NoError(t, err)
	require.NotEmpty(t, res.Text)
}
