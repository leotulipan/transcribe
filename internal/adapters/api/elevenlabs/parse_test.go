package elevenlabs

import (
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

func TestParse_Fixture(t *testing.T) {
	data, err := os.ReadFile("../../../../testdata/elevenlabs_sample.json")
	require.NoError(t, err)

	r, err := parse(data, "scribe_v1")
	require.NoError(t, err)

	require.Equal(t, "Hello world", r.Text)
	require.Equal(t, "eng", r.Language)
	// Words: only "word" type entries (spacing filtered)
	require.Len(t, r.Words, 2)
	require.Equal(t, "Hello", r.Words[0].Text)
	require.Equal(t, 100*time.Millisecond, r.Words[0].Start)
	require.Equal(t, 600*time.Millisecond, r.Words[0].End)
	require.Equal(t, "world", r.Words[1].Text)
	require.Equal(t, 700*time.Millisecond, r.Words[1].Start)
	require.Equal(t, 1200*time.Millisecond, r.Words[1].End)
	require.Equal(t, domain.ProviderElevenLabs, r.Provider)
	require.Equal(t, "scribe_v1", r.Model)
	require.NotEmpty(t, r.RawJSON)
}

func TestParse_BadJSON(t *testing.T) {
	_, err := parse([]byte("{bad json}"), "scribe_v1")
	require.Error(t, err)
}

func TestNormalizeSpeakerID(t *testing.T) {
	cases := map[string]string{
		"speaker_0": "0",
		"speaker_1": "1",
		"SPEAKER_2": "2",
		"speaker 3": "3",
		"A":         "A",
		"":          "",
		"  B  ":     "B",
	}
	for in, want := range cases {
		if got := normalizeSpeakerID(in); got != want {
			t.Errorf("normalizeSpeakerID(%q) = %q, want %q", in, got, want)
		}
	}
}

// TestParse_NormalizesElevenLabsSpeakerIDs verifies the real API "speaker_N"
// shape is normalized to bare tokens on Word.Speaker and the Speakers slice.
func TestParse_NormalizesElevenLabsSpeakerIDs(t *testing.T) {
	raw := []byte(`{"language_code":"deu","text":"a b",
		"words":[
			{"text":"a","start":0.0,"end":0.1,"type":"word","speaker_id":"speaker_0"},
			{"text":" ","start":0.1,"end":0.2,"type":"spacing"},
			{"text":"b","start":0.2,"end":0.3,"type":"word","speaker_id":"speaker_1"}
		]}`)
	r, err := parse(raw, "scribe_v1")
	require.NoError(t, err)
	require.Equal(t, "0", r.Words[0].Speaker)
	require.Equal(t, "1", r.Words[1].Speaker)
	ids := []string{r.Speakers[0].ID, r.Speakers[1].ID}
	require.ElementsMatch(t, []string{"0", "1"}, ids)
}
