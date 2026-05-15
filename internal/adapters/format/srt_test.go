package format

import (
	"os"
	"path/filepath"
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
	require.NoError(t, NewSRT().Write(res, dst))

	got, err := os.ReadFile(dst)
	require.NoError(t, err)

	golden, err := os.ReadFile("testdata/sample.srt.golden")
	require.NoError(t, err)
	require.Equal(t, string(golden), string(got))
}
