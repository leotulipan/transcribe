package elevenlabs

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestClient_DiscoverModels(t *testing.T) {
	t.Run("filters TTS models, keeps scribe_*", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			require.Equal(t, "/v1/models", r.URL.Path)
			require.Equal(t, "test-key", r.Header.Get("xi-api-key"))
			_, _ = w.Write([]byte(`[
				{"model_id":"eleven_multilingual_v2","name":"Eleven Multilingual v2"},
				{"model_id":"eleven_v3","name":"Eleven v3"},
				{"model_id":"scribe_v1","name":"Scribe v1"},
				{"model_id":"scribe_v2","name":"Scribe v2"}
			]`))
		}))
		defer srv.Close()
		c := NewWithEndpoint("test-key", srv.URL, http.DefaultClient)
		got, err := c.DiscoverModels(context.Background())
		require.NoError(t, err)
		// SortUnique returns alphabetical order; only scribe_* survive.
		require.Equal(t, []string{"scribe_v1", "scribe_v2"}, got)
	})

	t.Run("falls back to hardcoded list when zero scribe_* returned", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(`[
				{"model_id":"eleven_multilingual_v2","name":"Eleven Multilingual v2"},
				{"model_id":"eleven_v3","name":"Eleven v3"}
			]`))
		}))
		defer srv.Close()
		c := NewWithEndpoint("test-key", srv.URL, http.DefaultClient)
		got, err := c.DiscoverModels(context.Background())
		require.NoError(t, err)
		// Hardcoded fallback is scribe_v2, scribe_v1; SortUnique alphabetises.
		require.Equal(t, []string{"scribe_v1", "scribe_v2"}, got)
	})
}
