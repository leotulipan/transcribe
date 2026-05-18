package groq

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

func TestClient_DiscoverModels(t *testing.T) {
	t.Run("parses + sorts + dedupes", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			require.Equal(t, "GET", r.Method)
			require.Equal(t, "/openai/v1/models", r.URL.Path)
			require.Equal(t, "Bearer test-key", r.Header.Get("Authorization"))
			_, _ = w.Write([]byte(`{"data":[
				{"id":"whisper-large-v3"},
				{"id":"whisper-large-v3-turbo"},
				{"id":"whisper-large-v3"},
				{"id":""}
			]}`))
		}))
		defer srv.Close()
		c := NewWithEndpoint("test-key", srv.URL, http.DefaultClient)
		got, err := c.DiscoverModels(context.Background())
		require.NoError(t, err)
		require.Equal(t, []string{"whisper-large-v3", "whisper-large-v3-turbo"}, got)
	})
	t.Run("401 returns ErrProvider", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusUnauthorized)
		}))
		defer srv.Close()
		c := NewWithEndpoint("bad", srv.URL, http.DefaultClient)
		_, err := c.DiscoverModels(context.Background())
		var pe *domain.ErrProvider
		require.ErrorAs(t, err, &pe)
		require.Equal(t, 401, pe.StatusCode)
	})
}
