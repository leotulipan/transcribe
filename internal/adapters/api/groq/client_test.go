package groq

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
	fixture, err := os.ReadFile("../../../../testdata/groq_sample.json")
	require.NoError(t, err)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "POST", r.Method)
		require.Equal(t, "/openai/v1/audio/transcriptions", r.URL.Path)
		require.Equal(t, "Bearer test-key", r.Header.Get("Authorization"))
		require.NoError(t, r.ParseMultipartForm(32<<20))
		require.Equal(t, "whisper-large-v3", r.FormValue("model"))
		f, _, err := r.FormFile("file")
		require.NoError(t, err)
		defer f.Close()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(fixture)
	}))
	defer srv.Close()

	// tiny on-disk file the client can stream
	audioPath := filepath.Join(t.TempDir(), "tiny.mp3")
	require.NoError(t, os.WriteFile(audioPath, []byte("\xff\xfb\x90\x00"), 0o644))

	c := NewWithEndpoint("test-key", srv.URL, http.DefaultClient)
	require.Equal(t, domain.ProviderGroq, c.ID())
	res, err := c.Transcribe(context.Background(),
		domain.AudioFile{Path: audioPath, Container: "mp3", Codec: "mp3", SizeBytes: 4},
		ports.ProviderOpts{Model: "whisper-large-v3"},
	)
	require.NoError(t, err)
	require.Equal(t, "Hello world this is a test.", res.Text)
}
