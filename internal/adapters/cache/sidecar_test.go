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

func writeEmpty(path string) error { return writeBytes(path, []byte("")) }
func writeBytes(path string, b []byte) error {
	return osWriteFile(path, b)
}
