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

func TestSRT_Write_Golden(t *testing.T) {
	res := &domain.Result{
		Words: []domain.Word{
			{Text: "Hello",  Start: 1100 * time.Millisecond, End: 1500 * time.Millisecond},
			{Text: "world",  Start: 1600 * time.Millisecond, End: 2100 * time.Millisecond},
			{Text: "this",   Start: 2300 * time.Millisecond, End: 2600 * time.Millisecond},
			{Text: "is",     Start: 2700 * time.Millisecond, End: 2900 * time.Millisecond},
			{Text: "a",      Start: 3000 * time.Millisecond, End: 3100 * time.Millisecond},
			{Text: "test",   Start: 3200 * time.Millisecond, End: 3700 * time.Millisecond},
			// forced break by gap
			{Text: "second", Start: 8000 * time.Millisecond, End: 8500 * time.Millisecond},
			{Text: "block",  Start: 8600 * time.Millisecond, End: 9000 * time.Millisecond},
		},
	}
	dir := t.TempDir()
	dst := filepath.Join(dir, "out.srt")
	require.NoError(t, NewSRT().Write(res, dst, domain.WriteOpts{}))

	got, err := os.ReadFile(dst)
	require.NoError(t, err)

	golden, err := os.ReadFile("testdata/sample.srt.golden")
	require.NoError(t, err)
	require.Equal(t, string(golden), string(got))
}

func TestSRT_Write_StartHour(t *testing.T) {
	// Two words 1s apart, both inside the first 2 seconds.
	// With StartHour=1 every timecode shifts by +1h.
	res := &domain.Result{
		Words: []domain.Word{
			{Text: "hello", Start: 500 * time.Millisecond, End: 900 * time.Millisecond},
			{Text: "world", Start: 1000 * time.Millisecond, End: 1500 * time.Millisecond},
		},
	}
	dir := t.TempDir()
	dst := filepath.Join(dir, "out.srt")
	require.NoError(t, NewSRT().Write(res, dst, domain.WriteOpts{StartHour: 1}))

	got, err := os.ReadFile(dst)
	require.NoError(t, err)

	content := string(got)
	require.Contains(t, content, "01:00:00,500 --> 01:00:01,500", "StartHour=1 must shift all timecodes by +1h")
}

func TestSRT_Write_WordsPerSubtitle(t *testing.T) {
	// 6 tightly-spaced words; default grouping (7) puts them in one block,
	// but WordsPerSubtitle=2 must produce 3 blocks.
	words := []domain.Word{
		{Text: "one",   Start: 100 * time.Millisecond, End: 200 * time.Millisecond},
		{Text: "two",   Start: 300 * time.Millisecond, End: 400 * time.Millisecond},
		{Text: "three", Start: 500 * time.Millisecond, End: 600 * time.Millisecond},
		{Text: "four",  Start: 700 * time.Millisecond, End: 800 * time.Millisecond},
		{Text: "five",  Start: 900 * time.Millisecond, End: 1000 * time.Millisecond},
		{Text: "six",   Start: 1100 * time.Millisecond, End: 1200 * time.Millisecond},
	}
	res := &domain.Result{Words: words}
	dir := t.TempDir()

	// Default (7): all 6 words fit in one block.
	dstDefault := filepath.Join(dir, "default.srt")
	require.NoError(t, NewSRT().Write(res, dstDefault, domain.WriteOpts{}))
	gotDefault, _ := os.ReadFile(dstDefault)
	defaultBlocks := strings.Count(string(gotDefault), "\n\n")
	require.Equal(t, 1, defaultBlocks, "6 words with default limit should be one block")

	// WordsPerSubtitle=2: 6 words → 3 blocks.
	dstTwo := filepath.Join(dir, "two.srt")
	require.NoError(t, NewSRT().Write(res, dstTwo, domain.WriteOpts{WordsPerSubtitle: 2}))
	gotTwo, _ := os.ReadFile(dstTwo)
	twoBlocks := strings.Count(string(gotTwo), "\n\n")
	require.Equal(t, 3, twoBlocks, "6 words with WordsPerSubtitle=2 should produce 3 blocks")
}

func TestDaVinci_Write_StartHour(t *testing.T) {
	res := &domain.Result{
		Words: []domain.Word{
			{Text: "hello", Start: 500 * time.Millisecond, End: 900 * time.Millisecond},
			{Text: "world", Start: 1000 * time.Millisecond, End: 1500 * time.Millisecond},
		},
	}
	dir := t.TempDir()
	dst := filepath.Join(dir, "out.davinci.srt")
	require.NoError(t, NewDaVinci().Write(res, dst, domain.WriteOpts{StartHour: 2}))

	got, err := os.ReadFile(dst)
	require.NoError(t, err)

	content := string(got)
	require.Contains(t, content, "02:00:00,500 --> 02:00:01,500", "StartHour=2 must shift DaVinci timecodes by +2h")
}
