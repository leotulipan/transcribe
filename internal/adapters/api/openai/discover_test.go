package openai

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestClient_DiscoverModels(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/v1/models", r.URL.Path)
		require.Equal(t, "Bearer test-key", r.Header.Get("Authorization"))
		_, _ = w.Write([]byte(`{"data":[{"id":"gpt-4o-mini-transcribe"},{"id":"whisper-1"}]}`))
	}))
	defer srv.Close()
	c := NewWithEndpoint("test-key", srv.URL, http.DefaultClient)
	got, err := c.DiscoverModels(context.Background())
	require.NoError(t, err)
	require.Equal(t, []string{"gpt-4o-mini-transcribe", "whisper-1"}, got)
}
