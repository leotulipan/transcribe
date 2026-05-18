package gemini

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestClient_DiscoverModels(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/v1beta/models", r.URL.Path)
		require.Equal(t, "test-key", r.Header.Get("x-goog-api-key"))
		_, _ = w.Write([]byte(`{"models":[
			{"name":"models/gemini-2.5-flash"},
			{"name":"models/gemini-2.5-pro"}
		]}`))
	}))
	defer srv.Close()
	c := NewWithEndpoint("test-key", srv.URL, http.DefaultClient)
	got, err := c.DiscoverModels(context.Background())
	require.NoError(t, err)
	require.Equal(t, []string{"gemini-2.5-flash", "gemini-2.5-pro"}, got)
}
