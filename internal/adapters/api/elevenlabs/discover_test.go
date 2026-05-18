package elevenlabs

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
		require.Equal(t, "test-key", r.Header.Get("xi-api-key"))
		_, _ = w.Write([]byte(`[
			{"model_id":"eleven_multilingual_v2","name":"Eleven Multilingual v2"},
			{"model_id":"scribe_v1","name":"Scribe v1"}
		]`))
	}))
	defer srv.Close()
	c := NewWithEndpoint("test-key", srv.URL, http.DefaultClient)
	got, err := c.DiscoverModels(context.Background())
	require.NoError(t, err)
	require.Equal(t, []string{"eleven_multilingual_v2", "scribe_v1"}, got)
}
