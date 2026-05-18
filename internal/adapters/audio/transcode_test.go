package audio

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/ports"
)

func TestTranscode_MP3(t *testing.T) {
	skipIfNoFFmpeg(t)
	f, err := New("", "", nopLogger{})
	require.NoError(t, err)
	src, err := f.Probe("../../../testdata/short-sample.mp3")
	require.NoError(t, err)

	out, err := f.Transcode(context.Background(), src, ports.TargetFormat{
		Codec: "mp3", Bitrate: "64k",
	}, t.TempDir())
	require.NoError(t, err)
	require.True(t, out.Complete)
	require.Equal(t, "mp3", out.Codec)
	require.Equal(t, "mp3", out.Container)
	require.Greater(t, out.SizeBytes, int64(0))
}

func TestTranscode_FLAC(t *testing.T) {
	skipIfNoFFmpeg(t)
	f, err := New("", "", nopLogger{})
	require.NoError(t, err)
	src, err := f.Probe("../../../testdata/short-sample.mp3")
	require.NoError(t, err)

	out, err := f.Transcode(context.Background(), src, ports.TargetFormat{Codec: "flac"}, t.TempDir())
	require.NoError(t, err)
	require.Equal(t, "flac", out.Container)
	require.Equal(t, "flac", out.Codec)
}
