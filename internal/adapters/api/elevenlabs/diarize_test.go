package elevenlabs

import (
	"context"
	"mime"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

// readMultipartField parses the multipart body from a request and returns the
// value of the named field.
func readMultipartField(t *testing.T, r *http.Request, field string) string {
	t.Helper()
	_, params, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
	require.NoError(t, err)
	mr := multipart.NewReader(r.Body, params["boundary"])
	form, err := mr.ReadForm(1 << 20)
	require.NoError(t, err)
	vals := form.Value[field]
	require.NotEmpty(t, vals, "field %q not found in multipart", field)
	return vals[0]
}

func TestDiarize_RequestPayloadIncludesDiarizeFlag(t *testing.T) {
	fixture, err := os.ReadFile("../../../../testdata/elevenlabs_diarize.json")
	require.NoError(t, err)

	var gotDiarize string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotDiarize = readMultipartField(t, r, "diarize")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(fixture)
	}))
	defer srv.Close()

	audioPath := filepath.Join(t.TempDir(), "tiny.mp3")
	require.NoError(t, os.WriteFile(audioPath, []byte("\xff\xfb\x90\x00"), 0o644))

	c := NewWithEndpoint("test-key", srv.URL, http.DefaultClient)
	_, err = c.Transcribe(context.Background(),
		domain.AudioFile{Path: audioPath, Container: "mp3", Codec: "mp3", SizeBytes: 4},
		ports.ProviderOpts{Model: "scribe_v1", SpeakerLabels: true},
	)
	require.NoError(t, err)
	require.Equal(t, "true", gotDiarize, "multipart diarize field must be \"true\" when SpeakerLabels=true")
}

func TestDiarize_ParsePopulatesSpeakerOnWords(t *testing.T) {
	fixture, err := os.ReadFile("../../../../testdata/elevenlabs_diarize.json")
	require.NoError(t, err)

	r, err := parse(fixture, "scribe_v1")
	require.NoError(t, err)

	require.Len(t, r.Words, 2)
	require.Equal(t, "A", r.Words[0].Speaker, "first word speaker should be A")
	require.Equal(t, "B", r.Words[1].Speaker, "second word speaker should be B")

	// Speakers slice should contain unique speakers, de-duped.
	ids := make([]string, len(r.Speakers))
	for i, s := range r.Speakers {
		ids[i] = s.ID
	}
	require.ElementsMatch(t, []string{"A", "B"}, ids)
}

func TestDiarize_OptedOutLeavesSpeakerEmpty(t *testing.T) {
	// When SpeakerLabels=false the multipart body must send diarize=false,
	// and the parser leaves Word.Speaker empty (no speaker_id in the response).
	fixture, err := os.ReadFile("../../../../testdata/elevenlabs_sample.json")
	require.NoError(t, err)

	var gotDiarize string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotDiarize = readMultipartField(t, r, "diarize")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(fixture)
	}))
	defer srv.Close()

	audioPath := filepath.Join(t.TempDir(), "tiny.mp3")
	require.NoError(t, os.WriteFile(audioPath, []byte("\xff\xfb\x90\x00"), 0o644))

	c := NewWithEndpoint("test-key", srv.URL, http.DefaultClient)
	res, err := c.Transcribe(context.Background(),
		domain.AudioFile{Path: audioPath, Container: "mp3", Codec: "mp3", SizeBytes: 4},
		ports.ProviderOpts{Model: "scribe_v1", SpeakerLabels: false},
	)
	require.NoError(t, err)
	require.Equal(t, "false", gotDiarize, "multipart diarize field must be \"false\" when SpeakerLabels=false")

	for _, w := range res.Words {
		require.Empty(t, w.Speaker, "Word.Speaker must be empty when diarization not requested")
	}
	// The fixture has no speaker_id fields, so the strings should be empty.
	require.True(t, strings.Contains(string(fixture), "Hello"), "sanity: fixture loaded")
}
