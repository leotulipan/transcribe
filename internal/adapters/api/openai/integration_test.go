//go:build integration

package openai

import (
	"context"
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/integration"
	"github.com/leotulipan/transcribe/internal/ports"
)

func TestIntegration_OpenAI_CheckKey(t *testing.T) {
	key := integration.Key(t, domain.ProviderOpenAI)
	c := New(key, http.DefaultClient)
	require.NoError(t, c.CheckKey(context.Background()))
}

func TestIntegration_OpenAI_DiscoverModels(t *testing.T) {
	key := integration.Key(t, domain.ProviderOpenAI)
	c := New(key, http.DefaultClient)
	models, err := c.DiscoverModels(context.Background())
	require.NoError(t, err)
	require.NotEmpty(t, models)
	t.Logf("openai discovered %d models: %v", len(models), models)
}

func TestIntegration_OpenAI_Transcribe(t *testing.T) {
	key := integration.Key(t, domain.ProviderOpenAI)
	c := New(key, http.DefaultClient)
	res, err := c.Transcribe(context.Background(),
		domain.AudioFile{Path: "../../../../testdata/short-sample.mp3", Container: "mp3", Codec: "mp3"},
		ports.ProviderOpts{Model: c.DefaultModel(), Language: "en"},
	)
	require.NoError(t, err)
	require.NotEmpty(t, res.Text)
}
