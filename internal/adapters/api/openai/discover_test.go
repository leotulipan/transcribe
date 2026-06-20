package openai

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestClient_DiscoverModels(t *testing.T) {
	// OpenAI's /v1/models returns every model the key can access. Only the
	// speech-to-text ones (whisper*, *transcribe*) should survive.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/v1/models", r.URL.Path)
		require.Equal(t, "Bearer test-key", r.Header.Get("Authorization"))
		_, _ = w.Write([]byte(`{"data":[
			{"id":"gpt-4o-mini-transcribe"},
			{"id":"whisper-1"},
			{"id":"gpt-4o-transcribe"},
			{"id":"gpt-4o"},
			{"id":"text-embedding-3-small"},
			{"id":"tts-1"},
			{"id":"dall-e-3"}
		]}`))
	}))
	defer srv.Close()
	c := NewWithEndpoint("test-key", srv.URL, http.DefaultClient)
	got, err := c.DiscoverModels(context.Background())
	require.NoError(t, err)
	require.Equal(t, []string{"gpt-4o-mini-transcribe", "gpt-4o-transcribe", "whisper-1"}, got)
}

func TestClient_DiscoverModels_FallsBackWhenNoSTT(t *testing.T) {
	// If the key only has non-STT models, fall back to the hardcoded STT list
	// rather than showing an empty picker.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"data":[{"id":"gpt-4o"},{"id":"text-embedding-3-small"}]}`))
	}))
	defer srv.Close()
	c := NewWithEndpoint("test-key", srv.URL, http.DefaultClient)
	got, err := c.DiscoverModels(context.Background())
	require.NoError(t, err)
	require.Equal(t, Models(), got)
}
