package audio

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestCopyAudio_StreamCopiesIntoDerivedContainer(t *testing.T) {
	skipIfNoFFmpeg(t)
	f, err := New("", "", nopLogger{})
	require.NoError(t, err)

	src, err := f.Probe("../../../testdata/short-sample.mp3")
	require.NoError(t, err)

	workDir := t.TempDir()
	out, err := f.CopyAudio(context.Background(), src, workDir)
	require.NoError(t, err)
	require.True(t, out.IsTemp)
	require.True(t, out.Complete)
	require.Equal(t, "mp3", out.Container)
	require.Equal(t, "mp3", out.Codec)
	require.Greater(t, out.SizeBytes, int64(0))

	_, err = os.Stat(out.Path)
	require.NoError(t, err)
	require.Equal(t, filepath.Dir(out.Path), workDir)

	// partial file must not linger
	_, err = os.Stat(partialPath(out.Path))
	require.ErrorIs(t, err, os.ErrNotExist)
}
