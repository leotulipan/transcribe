package assemblyai

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

func TestClient_CheckKey(t *testing.T) {
	t.Run("200 ok", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			require.Equal(t, "GET", r.Method)
			require.Equal(t, "/v2/transcript", r.URL.Path)
			require.Equal(t, "1", r.URL.Query().Get("limit"))
			require.Equal(t, "test-key", r.Header.Get("Authorization"))
			_, _ = w.Write([]byte(`{"transcripts":[]}`))
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
		var pe *domain.ErrProvider
		require.ErrorAs(t, err, &pe)
		require.Equal(t, 401, pe.StatusCode)
	})
}

func TestClient_TranscribeThreeSteps(t *testing.T) {
	// Speed up polling for the test.
	origInterval := pollInterval
	pollInterval = 1 * time.Millisecond
	defer func() { pollInterval = origInterval }()

	fixture, err := os.ReadFile("../../../../testdata/assemblyai_sample.json")
	require.NoError(t, err)

	var pollCount atomic.Int32

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/v2/upload":
			require.Equal(t, "POST", r.Method)
			require.Equal(t, "test-key", r.Header.Get("Authorization"))
			require.Equal(t, "application/octet-stream", r.Header.Get("Content-Type"))
			// Return the upload URL pointing back at this test server.
			_, _ = w.Write([]byte(`{"upload_url": "` + "http://" + r.Host + `/audio/x` + `"}`))
		case "/v2/transcript":
			require.Equal(t, "POST", r.Method)
			require.Equal(t, "test-key", r.Header.Get("Authorization"))
			_, _ = w.Write([]byte(`{"id": "abc", "status": "queued"}`))
		case "/v2/transcript/abc":
			require.Equal(t, "GET", r.Method)
			n := pollCount.Add(1)
			if n < 2 {
				// Simulate processing state on first poll.
				_, _ = w.Write([]byte(`{"id": "abc", "status": "processing"}`))
			} else {
				_, _ = w.Write(fixture)
			}
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	audioPath := filepath.Join(t.TempDir(), "tiny.mp3")
	require.NoError(t, os.WriteFile(audioPath, []byte("\xff\xfb\x90\x00"), 0o644))

	c := NewWithEndpoint("test-key", srv.URL, http.DefaultClient)
	require.Equal(t, domain.ProviderAssemblyAI, c.ID())
	res, err := c.Transcribe(context.Background(),
		domain.AudioFile{Path: audioPath, Container: "mp3", Codec: "mp3", SizeBytes: 4},
		ports.ProviderOpts{Model: "best"},
	)
	require.NoError(t, err)
	require.Equal(t, "Hello world", res.Text)
	require.Len(t, res.Words, 2)
	require.GreaterOrEqual(t, pollCount.Load(), int32(2))
}

func TestClient_UploadFailureReturnsError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v2/upload" {
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = w.Write([]byte(`{"error": "unauthorized"}`))
		}
	}))
	defer srv.Close()

	audioPath := filepath.Join(t.TempDir(), "tiny.mp3")
	require.NoError(t, os.WriteFile(audioPath, []byte("\xff\xfb\x90\x00"), 0o644))

	c := NewWithEndpoint("bad-key", srv.URL, http.DefaultClient)
	_, err := c.Transcribe(context.Background(),
		domain.AudioFile{Path: audioPath, Container: "mp3", Codec: "mp3", SizeBytes: 4},
		ports.ProviderOpts{Model: "best"},
	)
	require.Error(t, err)
	var pe *domain.ErrProvider
	require.ErrorAs(t, err, &pe)
	require.Equal(t, domain.ProviderAssemblyAI, pe.Provider)
}
