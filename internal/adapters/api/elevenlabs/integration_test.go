//go:build integration

package elevenlabs

import (
	"context"
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/integration"
	"github.com/leotulipan/transcribe/internal/ports"
)

func TestIntegration_ElevenLabs_CheckKey(t *testing.T) {
	key := integration.Key(t, domain.ProviderElevenLabs)
	c := New(key, http.DefaultClient)
	require.NoError(t, c.CheckKey(context.Background()))
}

func TestIntegration_ElevenLabs_Transcribe(t *testing.T) {
	key := integration.Key(t, domain.ProviderElevenLabs)
	c := New(key, http.DefaultClient)
	res, err := c.Transcribe(context.Background(),
		domain.AudioFile{Path: "../../../../testdata/short-sample.mp3", Container: "mp3", Codec: "mp3"},
		ports.ProviderOpts{Model: c.DefaultModel(), Language: "en"},
	)
	require.NoError(t, err)
	require.NotEmpty(t, res.Text)
}
