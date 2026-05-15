package audio

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestExtractAudio_ProducesMonoPCMWav(t *testing.T) {
	skipIfNoFFmpeg(t)
	f, err := New("", "", nopLogger{})
	require.NoError(t, err)

	workDir := t.TempDir()
	out, err := f.ExtractAudio(context.Background(), "../../../testdata/short-sample.mp3", workDir)
	require.NoError(t, err)
	require.True(t, out.IsTemp)
	require.True(t, out.Complete)
	require.Equal(t, "wav", out.Container)
	require.Contains(t, out.Codec, "pcm")
}
