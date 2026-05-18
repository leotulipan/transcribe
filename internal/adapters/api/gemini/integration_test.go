//go:build integration

package gemini

import (
	"context"
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/integration"
	"github.com/leotulipan/transcribe/internal/ports"
)

func TestIntegration_Gemini_CheckKey(t *testing.T) {
	key := integration.Key(t, domain.ProviderGemini)
	c := New(key, http.DefaultClient)
	require.NoError(t, c.CheckKey(context.Background()))
}

func TestIntegration_Gemini_DiscoverModels(t *testing.T) {
	key := integration.Key(t, domain.ProviderGemini)
	c := New(key, http.DefaultClient)
	models, err := c.DiscoverModels(context.Background())
	require.NoError(t, err)
	require.NotEmpty(t, models)
	t.Logf("gemini discovered %d models: %v", len(models), models)
}

func TestIntegration_Gemini_Transcribe(t *testing.T) {
	key := integration.Key(t, domain.ProviderGemini)
	c := New(key, http.DefaultClient)
	res, err := c.Transcribe(context.Background(),
		domain.AudioFile{Path: "../../../../testdata/short-sample.mp3", Container: "mp3", Codec: "mp3"},
		ports.ProviderOpts{Model: c.DefaultModel()},
	)
	require.NoError(t, err)
	require.NotEmpty(t, res.Text)
}
