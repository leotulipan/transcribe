package format

import (
	"os"
	"path/filepath"
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
