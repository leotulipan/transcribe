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

func TestDaVinci_Write_Golden(t *testing.T) {
	res := &domain.Result{
		Words: []domain.Word{
			{Text: "Wir",       Start: 1100 * time.Millisecond, End: 1300 * time.Millisecond},
			{Text: "testen",    Start: 1350 * time.Millisecond, End: 1700 * time.Millisecond},
			{Text: "ÄHM",       Start: 1750 * time.Millisecond, End: 1900 * time.Millisecond}, // filler (uppercase)
			{Text: "das",       Start: 2000 * time.Millisecond, End: 2200 * time.Millisecond},
			{Text: "Skript",    Start: 2300 * time.Millisecond, End: 2700 * time.Millisecond},
			{Text: "(...)",     Start: 2700 * time.Millisecond, End: 4500 * time.Millisecond}, // pause marker
			{Text: "Nochmal",   Start: 4500 * time.Millisecond, End: 5000 * time.Millisecond},
		},
	}
	dir := t.TempDir()
	dst := filepath.Join(dir, "out.davinci.srt")
	require.NoError(t, NewDaVinci().Write(res, dst, domain.WriteOpts{}))

	got, err := os.ReadFile(dst)
	require.NoError(t, err)

	golden, err := os.ReadFile("testdata/sample.davinci.srt.golden")
	require.NoError(t, err)
	require.Equal(t, string(golden), string(got))
}

func TestDaVinci_PauseMarkerHasNoSpeakerPrefixWhenLabelsEnabled(t *testing.T) {
	res := &domain.Result{
		Words: []domain.Word{
			{Text: "Hello", Start: 0 * time.Millisecond, End: 500 * time.Millisecond, Speaker: "A"},
			{Text: "(...)", Start: 500 * time.Millisecond, End: 2000 * time.Millisecond, Speaker: ""},
			{Text: "World", Start: 2000 * time.Millisecond, End: 2500 * time.Millisecond, Speaker: "A"},
		},
	}
	dir := t.TempDir()
	dst := filepath.Join(dir, "out.davinci.srt")
	require.NoError(t, NewDaVinci().Write(res, dst, domain.WriteOpts{SpeakerLabels: true}))

	got, err := os.ReadFile(dst)
	require.NoError(t, err)
	gotStr := string(got)

	// Pause marker block (block 2) should not have speaker prefix
	require.Contains(t, gotStr, "[Speaker A]: Hello")
	require.Contains(t, gotStr, "(...)")
	require.NotContains(t, gotStr, "[Speaker ]: (...)")
	// Verify structure: pause marker block exists without speaker prefix
	lines := strings.Split(gotStr, "\n")
	pauseBlockFound := false
	for i, line := range lines {
		if line == "(...)" {
			pauseBlockFound = true
			// The line before should not be a speaker prefix
			if i > 0 {
				require.NotContains(t, lines[i-1], "[Speaker")
			}
		}
	}
	require.True(t, pauseBlockFound, "pause marker block not found in output")
}
