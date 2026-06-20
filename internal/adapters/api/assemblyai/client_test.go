package assemblyai

import (
	"context"
	"encoding/json"
	"io"
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

// captureTranscriptPayload starts a test server that captures the JSON body
// sent to /v2/transcript and returns a minimal transcript response so the
// client can proceed. The captured payload is written to *got.
func captureTranscriptPayload(t *testing.T, got *map[string]interface{}) *httptest.Server {
	t.Helper()
	origInterval := pollInterval
	pollInterval = 1 * time.Millisecond
	t.Cleanup(func() { pollInterval = origInterval })

	fixture, err := os.ReadFile("../../../../testdata/assemblyai_sample.json")
	require.NoError(t, err)

	var polled atomic.Bool
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/v2/upload":
			_, _ = w.Write([]byte(`{"upload_url":"http://` + r.Host + `/audio/x"}`))
		case "/v2/transcript":
			body, _ := io.ReadAll(r.Body)
			_ = json.Unmarshal(body, got)
			_, _ = w.Write([]byte(`{"id":"abc","status":"queued"}`))
		case "/v2/transcript/abc":
			if polled.Swap(true) {
				_, _ = w.Write(fixture)
			} else {
				_, _ = w.Write([]byte(`{"id":"abc","status":"processing"}`))
			}
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
}

func TestAssemblyAI_RequestIncludesNumSpeakers(t *testing.T) {
	var got map[string]interface{}
	srv := captureTranscriptPayload(t, &got)
	defer srv.Close()

	audioPath := filepath.Join(t.TempDir(), "tiny.mp3")
	require.NoError(t, os.WriteFile(audioPath, []byte("\xff\xfb\x90\x00"), 0o644))

	c := NewWithEndpoint("test-key", srv.URL, http.DefaultClient)
	_, err := c.Transcribe(context.Background(),
		domain.AudioFile{Path: audioPath, Container: "mp3", Codec: "mp3", SizeBytes: 4},
		ports.ProviderOpts{Model: "best", SpeakerLabels: true, NumSpeakers: 3},
	)
	require.NoError(t, err)
	require.Equal(t, true, got["speaker_labels"], "speaker_labels must be true")
	require.InDelta(t, float64(3), got["speakers_expected"], 0.001, "speakers_expected must be 3")
}

func TestAssemblyAI_RequestIncludesKeyTerms(t *testing.T) {
	var got map[string]interface{}
	srv := captureTranscriptPayload(t, &got)
	defer srv.Close()

	audioPath := filepath.Join(t.TempDir(), "tiny.mp3")
	require.NoError(t, os.WriteFile(audioPath, []byte("\xff\xfb\x90\x00"), 0o644))

	c := NewWithEndpoint("test-key", srv.URL, http.DefaultClient)
	_, err := c.Transcribe(context.Background(),
		domain.AudioFile{Path: audioPath, Container: "mp3", Codec: "mp3", SizeBytes: 4},
		ports.ProviderOpts{Model: "best", KeyTerms: []string{"foo", "bar"}},
	)
	require.NoError(t, err)
	raw, ok := got["keyterms_prompt"].([]interface{})
	require.True(t, ok, "keyterms_prompt must be an array")
	require.Len(t, raw, 2)
	require.Equal(t, "foo", raw[0])
	require.Equal(t, "bar", raw[1])
}

func TestAssemblyAI_RequestIncludesSpeechModels(t *testing.T) {
	var got map[string]interface{}
	srv := captureTranscriptPayload(t, &got)
	defer srv.Close()

	audioPath := filepath.Join(t.TempDir(), "tiny.mp3")
	require.NoError(t, os.WriteFile(audioPath, []byte("\xff\xfb\x90\x00"), 0o644))

	c := NewWithEndpoint("test-key", srv.URL, http.DefaultClient)
	_, err := c.Transcribe(context.Background(),
		domain.AudioFile{Path: audioPath, Container: "mp3", Codec: "mp3", SizeBytes: 4},
		ports.ProviderOpts{Model: "best", SpeechModels: []string{"universal-3-pro", "universal-2"}},
	)
	require.NoError(t, err)
	raw, ok := got["speech_models"].([]interface{})
	require.True(t, ok, "speech_models must be an array")
	require.Len(t, raw, 2)
	require.Equal(t, "universal-3-pro", raw[0])
	require.Equal(t, "universal-2", raw[1])
}

func TestAssemblyAI_SpeechModelsDefaultsToFallback(t *testing.T) {
	// With no explicit fallback list, the selected model is sent followed by the
	// universal-2 fallback, and the deprecated singular `speech_model` is absent.
	var got map[string]interface{}
	srv := captureTranscriptPayload(t, &got)
	defer srv.Close()

	audioPath := filepath.Join(t.TempDir(), "tiny.mp3")
	require.NoError(t, os.WriteFile(audioPath, []byte("\xff\xfb\x90\x00"), 0o644))

	c := NewWithEndpoint("test-key", srv.URL, http.DefaultClient)
	_, err := c.Transcribe(context.Background(),
		domain.AudioFile{Path: audioPath, Container: "mp3", Codec: "mp3", SizeBytes: 4},
		ports.ProviderOpts{Model: "universal-3-pro"},
	)
	require.NoError(t, err)

	_, hasSingular := got["speech_model"]
	require.False(t, hasSingular, "deprecated singular speech_model must not be sent")

	raw, ok := got["speech_models"].([]interface{})
	require.True(t, ok, "speech_models must be an array")
	require.Equal(t, []interface{}{"universal-3-pro", "universal-2"}, raw)
}

func TestAssemblyAI_SpeechModelsNoDuplicateFallback(t *testing.T) {
	// When the selected model already is the fallback, it must not be duplicated.
	var got map[string]interface{}
	srv := captureTranscriptPayload(t, &got)
	defer srv.Close()

	audioPath := filepath.Join(t.TempDir(), "tiny.mp3")
	require.NoError(t, os.WriteFile(audioPath, []byte("\xff\xfb\x90\x00"), 0o644))

	c := NewWithEndpoint("test-key", srv.URL, http.DefaultClient)
	_, err := c.Transcribe(context.Background(),
		domain.AudioFile{Path: audioPath, Container: "mp3", Codec: "mp3", SizeBytes: 4},
		ports.ProviderOpts{Model: "universal-2"},
	)
	require.NoError(t, err)
	raw, ok := got["speech_models"].([]interface{})
	require.True(t, ok)
	require.Equal(t, []interface{}{"universal-2"}, raw)
}

func TestAssemblyAI_NoSpeakersExpectedWhenSpeakerLabelsFalse(t *testing.T) {
	var got map[string]interface{}
	srv := captureTranscriptPayload(t, &got)
	defer srv.Close()

	audioPath := filepath.Join(t.TempDir(), "tiny.mp3")
	require.NoError(t, os.WriteFile(audioPath, []byte("\xff\xfb\x90\x00"), 0o644))

	c := NewWithEndpoint("test-key", srv.URL, http.DefaultClient)
	_, err := c.Transcribe(context.Background(),
		domain.AudioFile{Path: audioPath, Container: "mp3", Codec: "mp3", SizeBytes: 4},
		ports.ProviderOpts{Model: "best", SpeakerLabels: false, NumSpeakers: 3},
	)
	require.NoError(t, err)
	_, hasSpeakersExpected := got["speakers_expected"]
	require.False(t, hasSpeakersExpected, "speakers_expected must not appear when SpeakerLabels=false")
	_, hasSpeakerLabels := got["speaker_labels"]
	require.False(t, hasSpeakerLabels, "speaker_labels must not appear when SpeakerLabels=false")
}
