package elevenlabs

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

func TestClient_TranscribePostsAndParses(t *testing.T) {
	fixture, err := os.ReadFile("../../../../testdata/elevenlabs_sample.json")
	require.NoError(t, err)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "POST", r.Method)
		require.Equal(t, "/v1/speech-to-text", r.URL.Path)
		require.Equal(t, "test-key", r.Header.Get("xi-api-key"))
		require.NoError(t, r.ParseMultipartForm(32<<20))
		require.Equal(t, "scribe_v1", r.FormValue("model_id"))
		require.Equal(t, "word", r.FormValue("timestamps_granularity"))
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
	require.Equal(t, domain.ProviderElevenLabs, c.ID())
	res, err := c.Transcribe(context.Background(),
		domain.AudioFile{Path: audioPath, Container: "mp3", Codec: "mp3", SizeBytes: 4},
		ports.ProviderOpts{Model: "scribe_v1"},
	)
	require.NoError(t, err)
	require.Equal(t, "Hello world", res.Text)
	require.Len(t, res.Words, 2)
}

func TestClient_NonOKReturnsError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"detail": "invalid key"}`))
	}))
	defer srv.Close()

	audioPath := filepath.Join(t.TempDir(), "tiny.mp3")
	require.NoError(t, os.WriteFile(audioPath, []byte("\xff\xfb\x90\x00"), 0o644))

	c := NewWithEndpoint("bad-key", srv.URL, http.DefaultClient)
	_, err := c.Transcribe(context.Background(),
		domain.AudioFile{Path: audioPath, Container: "mp3", Codec: "mp3", SizeBytes: 4},
		ports.ProviderOpts{Model: "scribe_v1"},
	)
	require.Error(t, err)
	var pe *domain.ErrProvider
	require.ErrorAs(t, err, &pe)
	require.Equal(t, domain.ProviderElevenLabs, pe.Provider)
}
