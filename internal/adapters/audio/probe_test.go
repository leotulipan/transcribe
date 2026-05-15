package audio

import (
	"os/exec"
	"testing"

	"github.com/stretchr/testify/require"
)

func skipIfNoFFmpeg(t *testing.T) {
	t.Helper()
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("ffmpeg not on PATH")
	}
	if _, err := exec.LookPath("ffprobe"); err != nil {
		t.Skip("ffprobe not on PATH")
	}
}

func TestProbe_ReportsFormatAndCodec(t *testing.T) {
	skipIfNoFFmpeg(t)
	f, err := New("", "", nopLogger{})
	require.NoError(t, err)
	af, err := f.Probe("../../../testdata/short-sample.mp3")
	require.NoError(t, err)
	require.Equal(t, "mp3", af.Codec)
	require.Greater(t, af.SizeBytes, int64(0))
	require.Greater(t, int64(af.Duration), int64(0))
}

type nopLogger struct{}

func (nopLogger) Debug(string, ...any) {}
func (nopLogger) Info(string, ...any)  {}
func (nopLogger) Warn(string, ...any)  {}
func (nopLogger) Error(string, ...any) {}
