package cli

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestSplitLabelValue(t *testing.T) {
	l, v, ok := splitLabelValue("Julia=C:/a/b.wav")
	require.True(t, ok)
	require.Equal(t, "Julia", l)
	require.Equal(t, "C:/a/b.wav", v)

	_, _, ok = splitLabelValue("noequals")
	require.False(t, ok)

	_, _, ok = splitLabelValue("=onlyvalue")
	require.False(t, ok)
}

func TestParseOffsets(t *testing.T) {
	got, err := parseOffsets([]string{"Gast=1.2s", "Julia=-500ms"})
	require.NoError(t, err)
	require.Equal(t, 1200*time.Millisecond, got["Gast"])
	require.Equal(t, -500*time.Millisecond, got["Julia"])

	_, err = parseOffsets([]string{"Gast=notaduration"})
	require.Error(t, err)
}

func TestCombinedBasePath(t *testing.T) {
	tracks := []speakerTrack{
		{label: "Julia", path: filepath.FromSlash("/pod/CON-259_julia.wav")},
		{label: "Gast", path: filepath.FromSlash("/pod/CON-259_guest.wav")},
	}
	got := combinedBasePath(tracks, "")
	require.Equal(t, filepath.FromSlash("/pod/CON-259_combined"), got)
}

func TestParseSpeakerTracks_RejectsDuplicateLabel(t *testing.T) {
	dir := t.TempDir()
	f := filepath.Join(dir, "a.wav")
	require.NoError(t, os.WriteFile(f, nil, 0o644))
	_, err := parseSpeakerTracks([]string{"A=" + f, "A=" + f})
	require.Error(t, err)
}
