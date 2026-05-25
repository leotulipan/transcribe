package cache

import (
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

func TestSidecar_RoundTrip(t *testing.T) {
	dir := t.TempDir()
	input := filepath.Join(dir, "talk.mp3")
	require.NoError(t, writeEmpty(input))

	c := New()
	res := &domain.Result{
		Provider: domain.ProviderGroq,
		Model:    "whisper-large-v3",
		Language: "en",
		Text:     "hello world",
		Duration: 5 * time.Second,
		Words: []domain.Word{
			{Text: "hello", Start: 0, End: 500 * time.Millisecond},
			{Text: "world", Start: 600 * time.Millisecond, End: time.Second},
		},
		RawJSON: []byte(`{"raw":"yes"}`),
	}
	require.NoError(t, c.Save(input, res))

	out, hit, err := c.Lookup(input, domain.ProviderGroq)
	require.NoError(t, err)
	require.True(t, hit)
	require.Equal(t, "hello world", out.Text)
	require.Equal(t, "whisper-large-v3", out.Model)
	require.Equal(t, 5*time.Second, out.Duration)
	require.Len(t, out.Words, 2)
	require.Equal(t, 500*time.Millisecond, out.Words[0].End)
}

func TestSidecar_MissReturnsFalse(t *testing.T) {
	dir := t.TempDir()
	input := filepath.Join(dir, "absent.mp3")
	require.NoError(t, writeEmpty(input))

	_, hit, err := New().Lookup(input, domain.ProviderGroq)
	require.NoError(t, err)
	require.False(t, hit)
}

func TestSidecar_UnknownSchemaIsMiss(t *testing.T) {
	dir := t.TempDir()
	input := filepath.Join(dir, "x.mp3")
	require.NoError(t, writeEmpty(input))

	side := sidecarPath(input, domain.ProviderGroq)
	require.NoError(t, writeBytes(side, []byte(`{"schema_version":999}`)))

	_, hit, err := New().Lookup(input, domain.ProviderGroq)
	require.NoError(t, err)
	require.False(t, hit, "unknown schema version should be treated as miss")
}

func TestSidecar_LoadFromFile_Roundtrip(t *testing.T) {
	dir := t.TempDir()
	input := filepath.Join(dir, "talk.mp3")
	require.NoError(t, writeEmpty(input))

	c := New()
	res := &domain.Result{
		Provider: domain.ProviderGroq,
		Model:    "whisper-large-v3",
		Language: "de",
		Text:     "hallo welt",
		Duration: 3 * time.Second,
		Words: []domain.Word{
			{Text: "hallo", Start: 0, End: 300 * time.Millisecond},
		},
		Segments: []domain.Segment{
			{Text: "hallo welt", Start: 0, End: 3 * time.Second, SpeakerID: "A"},
		},
	}
	require.NoError(t, c.Save(input, res))

	// Derive the path Save wrote, then read it back via LoadFromFile.
	jsonPath := sidecarPath(input, domain.ProviderGroq)
	got, err := c.LoadFromFile(jsonPath)
	require.NoError(t, err)
	require.Equal(t, "hallo welt", got.Text)
	require.Equal(t, "whisper-large-v3", got.Model)
	require.Equal(t, "de", got.Language)
	require.Equal(t, 3*time.Second, got.Duration)
	require.Len(t, got.Words, 1)
	require.Equal(t, 300*time.Millisecond, got.Words[0].End)
	require.Len(t, got.Segments, 1)
	require.Equal(t, "A", got.Segments[0].SpeakerID)
}

func TestSidecar_LoadFromFile_MissingFile(t *testing.T) {
	_, err := New().LoadFromFile(filepath.Join(t.TempDir(), "nonexistent.json"))
	require.Error(t, err)
}

func TestSidecar_LoadFromFile_BadSchema(t *testing.T) {
	dir := t.TempDir()
	jsonPath := filepath.Join(dir, "bad.json")
	require.NoError(t, writeBytes(jsonPath, []byte(`{"schema_version":999}`)))
	_, err := New().LoadFromFile(jsonPath)
	require.Error(t, err)
	require.Contains(t, err.Error(), "unsupported sidecar schema version")
}

func writeEmpty(path string) error { return writeBytes(path, []byte("")) }
func writeBytes(path string, b []byte) error {
	return osWriteFile(path, b)
}
