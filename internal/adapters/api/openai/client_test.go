package openai

import (
	"context"
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
			require.Equal(t, "/v1/models", r.URL.Path)
			require.Equal(t, "Bearer test-key", r.Header.Get("Authorization"))
			_, _ = w.Write([]byte(`{"data":[]}`))
		}))
		defer srv.Close()
		c := NewWithEndpoint("test-key", srv.URL, http.DefaultClient)
		require.NoError(t, c.CheckKey(context.Background()))
	})
	t.Run("401 returns ErrProvider", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusUnauthorized)
		}))
		defer srv.Close()
		c := NewWithEndpoint("bad", srv.URL, http.DefaultClient)
		err := c.CheckKey(context.Background())
		require.Error(t, err)
		var pe *domain.ErrProvider
		require.ErrorAs(t, err, &pe)
		require.Equal(t, 401, pe.StatusCode)
	})
}

func TestClient_TranscribePostsAndParses(t *testing.T) {
	fixture, err := os.ReadFile("../../../../testdata/openai_sample.json")
	require.NoError(t, err)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "POST", r.Method)
		require.Equal(t, "/v1/audio/transcriptions", r.URL.Path)
		require.Equal(t, "Bearer test-key", r.Header.Get("Authorization"))
		require.NoError(t, r.ParseMultipartForm(32<<20))
		require.Equal(t, "whisper-1", r.FormValue("model"))
		require.Equal(t, "verbose_json", r.FormValue("response_format"))
		f, _, err := r.FormFile("file")
		require.NoError(t, err)
		defer f.Close()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(fixture)
	}))
	defer srv.Close()

	audioPath := filepath.Join(t.TempDir(), "tiny.mp3")
	require.NoError(t, os.WriteFile(audioPath, []byte("\xff\xfb\x90\x00"), 0o644))

	c := NewWithEndpoint("test-key", srv.URL, http.DefaultClient)
	require.Equal(t, domain.ProviderOpenAI, c.ID())
	res, err := c.Transcribe(context.Background(),
		domain.AudioFile{Path: audioPath, Container: "mp3", Codec: "mp3", SizeBytes: 4},
		ports.ProviderOpts{Model: "whisper-1"},
	)
	require.NoError(t, err)
	require.Equal(t, "Goodbye cruel world this is OpenAI.", res.Text)
}

func TestClient_NonOKReturnsError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error": "invalid key"}`))
	}))
	defer srv.Close()

	audioPath := filepath.Join(t.TempDir(), "tiny.mp3")
	require.NoError(t, os.WriteFile(audioPath, []byte("\xff\xfb\x90\x00"), 0o644))

	c := NewWithEndpoint("bad-key", srv.URL, http.DefaultClient)
	_, err := c.Transcribe(context.Background(),
		domain.AudioFile{Path: audioPath, Container: "mp3", Codec: "mp3", SizeBytes: 4},
		ports.ProviderOpts{Model: "whisper-1"},
	)
	require.Error(t, err)
	var pe *domain.ErrProvider
	require.ErrorAs(t, err, &pe)
	require.Equal(t, domain.ProviderOpenAI, pe.Provider)
}
