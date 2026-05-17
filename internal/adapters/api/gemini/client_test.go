package gemini

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

func TestClient_CheckKey(t *testing.T) {
	t.Run("200 ok", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			require.Equal(t, "GET", r.Method)
			require.Equal(t, "/v1beta/models", r.URL.Path)
			require.Equal(t, "test-key", r.Header.Get("x-goog-api-key"))
			_, _ = w.Write([]byte(`{"models":[]}`))
		}))
		defer srv.Close()
		c := NewWithEndpoint("test-key", srv.URL, http.DefaultClient)
		require.NoError(t, c.CheckKey(context.Background()))
	})
	t.Run("403 returns ErrProvider", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusForbidden)
		}))
		defer srv.Close()
		c := NewWithEndpoint("bad", srv.URL, http.DefaultClient)
		err := c.CheckKey(context.Background())
		var pe *domain.ErrProvider
		require.ErrorAs(t, err, &pe)
		require.Equal(t, 403, pe.StatusCode)
	})
}

func TestClient_TranscribeTwoSteps(t *testing.T) {
	fixture, err := os.ReadFile("../../../../testdata/gemini_sample.json")
	require.NoError(t, err)

	const testFileURI = "https://files.googleapis.com/v1beta/files/test-file-id"

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.URL.Path == "/upload/v1beta/files":
			require.Equal(t, "POST", r.Method)
			require.Equal(t, "test-key", r.Header.Get("x-goog-api-key"))
			_, _ = fmt.Fprintf(w, `{"file": {"uri": "%s"}}`, testFileURI)
		case r.URL.Path == "/v1beta/models/gemini-2.0-flash:generateContent":
			require.Equal(t, "POST", r.Method)
			require.Equal(t, "test-key", r.Header.Get("x-goog-api-key"))
			_, _ = w.Write(fixture)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	audioPath := filepath.Join(t.TempDir(), "tiny.mp3")
	require.NoError(t, os.WriteFile(audioPath, []byte("\xff\xfb\x90\x00"), 0o644))

	c := NewWithEndpoint("test-key", srv.URL, http.DefaultClient)
	require.Equal(t, domain.ProviderGemini, c.ID())
	res, err := c.Transcribe(context.Background(),
		domain.AudioFile{Path: audioPath, Container: "mp3", Codec: "mp3", SizeBytes: 4},
		ports.ProviderOpts{Model: "gemini-2.0-flash"},
	)
	require.NoError(t, err)
	require.Equal(t, "Hello world", res.Text)
	require.Empty(t, res.Words)
	require.Empty(t, res.Segments)
}

func TestClient_UploadFailureReturnsError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/upload/v1beta/files" {
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = w.Write([]byte(`{"error": {"message": "API key not valid"}}`))
		}
	}))
	defer srv.Close()

	audioPath := filepath.Join(t.TempDir(), "tiny.mp3")
	require.NoError(t, os.WriteFile(audioPath, []byte("\xff\xfb\x90\x00"), 0o644))

	c := NewWithEndpoint("bad-key", srv.URL, http.DefaultClient)
	_, err := c.Transcribe(context.Background(),
		domain.AudioFile{Path: audioPath, Container: "mp3", Codec: "mp3", SizeBytes: 4},
		ports.ProviderOpts{Model: "gemini-2.0-flash"},
	)
	require.Error(t, err)
	var pe *domain.ErrProvider
	require.ErrorAs(t, err, &pe)
	require.Equal(t, domain.ProviderGemini, pe.Provider)
}
