package audio

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestChunk_SplitsUnderMaxBytes(t *testing.T) {
	skipIfNoFFmpeg(t)
	f, err := New("", "", nopLogger{})
	require.NoError(t, err)
	src, err := f.Probe("../../../testdata/short-sample.mp3")
	require.NoError(t, err)

	workDir := t.TempDir()
	// Force at least 2 chunks by halving the budget.
	budget := src.SizeBytes/2 + 1
	chunks, err := f.Chunk(context.Background(), src, budget, workDir)
	require.NoError(t, err)
	require.GreaterOrEqual(t, len(chunks), 2)
	for _, c := range chunks {
		require.True(t, c.Complete)
		info, err := os.Stat(c.Path)
		require.NoError(t, err)
		require.Equal(t, filepath.Dir(c.Path), workDir)
		require.LessOrEqual(t, info.Size(), budget+1024*1024) // 1MB slack for header re-emission
	}
}

func TestCleanup_DeletesIntermediateAndMeta(t *testing.T) {
	skipIfNoFFmpeg(t)
	f, err := New("", "", nopLogger{})
	require.NoError(t, err)
	workDir := t.TempDir()
	src, err := f.Probe("../../../testdata/short-sample.mp3")
	require.NoError(t, err)
	out, err := f.CopyAudio(context.Background(), src, workDir)
	require.NoError(t, err)
	require.NoError(t, WriteMeta(out.Path, MetaInfo{Operation: "copy"}))

	require.NoError(t, f.Cleanup(out))
	_, err = os.Stat(out.Path)
	require.ErrorIs(t, err, os.ErrNotExist)
	_, err = os.Stat(metaPath(out.Path))
	require.ErrorIs(t, err, os.ErrNotExist)
}
