package format

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

// TestSRT_SpeakerLabels verifies that [Speaker X]: prefixes are emitted when
// WriteOpts.SpeakerLabels is true, and suppressed when false.
func TestSRT_SpeakerLabels(t *testing.T) {
	res := &domain.Result{
		Words: []domain.Word{
			{Text: "Hello", Start: 100 * time.Millisecond, End: 500 * time.Millisecond, Speaker: "A"},
			{Text: "world", Start: 600 * time.Millisecond, End: 1000 * time.Millisecond, Speaker: "A"},
			// Gap forces a new block.
			{Text: "Goodbye", Start: 5000 * time.Millisecond, End: 5500 * time.Millisecond, Speaker: "B"},
		},
	}

	t.Run("labels_enabled", func(t *testing.T) {
		dir := t.TempDir()
		dst := filepath.Join(dir, "out.srt")
		require.NoError(t, NewSRT().Write(res, dst, domain.WriteOpts{SpeakerLabels: true}))
		got, _ := os.ReadFile(dst)
		s := string(got)
		require.True(t, strings.Contains(s, "[Speaker A]: Hello"), "first block must be prefixed with Speaker A")
		require.True(t, strings.Contains(s, "[Speaker B]: Goodbye"), "second block must be prefixed with Speaker B")
	})

	t.Run("labels_disabled", func(t *testing.T) {
		dir := t.TempDir()
		dst := filepath.Join(dir, "out.srt")
		require.NoError(t, NewSRT().Write(res, dst, domain.WriteOpts{SpeakerLabels: false}))
		got, _ := os.ReadFile(dst)
		s := string(got)
		require.False(t, strings.Contains(s, "[Speaker"), "no speaker prefix when SpeakerLabels=false")
	})
}

// TestDaVinci_SpeakerLabels verifies the same for the DaVinci writer.
func TestDaVinci_SpeakerLabels(t *testing.T) {
	res := &domain.Result{
		Words: []domain.Word{
			{Text: "Hello", Start: 100 * time.Millisecond, End: 500 * time.Millisecond, Speaker: "A"},
			{Text: "world", Start: 600 * time.Millisecond, End: 1000 * time.Millisecond, Speaker: "A"},
		},
	}

	t.Run("labels_enabled", func(t *testing.T) {
		dir := t.TempDir()
		dst := filepath.Join(dir, "out.davinci.srt")
		require.NoError(t, NewDaVinci().Write(res, dst, domain.WriteOpts{SpeakerLabels: true}))
		got, _ := os.ReadFile(dst)
		require.True(t, strings.Contains(string(got), "[Speaker A]: Hello"))
	})

	t.Run("labels_disabled", func(t *testing.T) {
		dir := t.TempDir()
		dst := filepath.Join(dir, "out.davinci.srt")
		require.NoError(t, NewDaVinci().Write(res, dst, domain.WriteOpts{SpeakerLabels: false}))
		got, _ := os.ReadFile(dst)
		require.False(t, strings.Contains(string(got), "[Speaker"))
	})
}
