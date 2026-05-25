package audio

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/ports"
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
	chunks, err := f.Chunk(context.Background(), src, budget, workDir, ports.ChunkOpts{})
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

func TestChunker_ChunkLengthOverride(t *testing.T) {
	skipIfNoFFmpeg(t)
	f, err := New("", "", nopLogger{})
	require.NoError(t, err)
	src, err := f.Probe("../../../testdata/short-sample.mp3")
	require.NoError(t, err)

	// Use a 1-second chunk length to force multiple chunks regardless of byte budget.
	workDir := t.TempDir()
	opts := ports.ChunkOpts{ChunkLengthSec: 1}
	chunks, err := f.Chunk(context.Background(), src, src.SizeBytes*10, workDir, opts)
	require.NoError(t, err)
	// Source is longer than 1 s, so we expect multiple chunks.
	require.Greater(t, len(chunks), 1, "ChunkLengthSec=1 should produce multiple chunks")
	// Each chunk's StartOffset should advance by ~1s.
	for i, c := range chunks {
		expected := time.Duration(i) * time.Second
		diff := c.StartOffset - expected
		if diff < 0 {
			diff = -diff
		}
		require.LessOrEqual(t, diff, 100*time.Millisecond, "chunk %d StartOffset should be ~%v", i, expected)
	}
}

func TestChunker_OverlapAddsToStartTime(t *testing.T) {
	skipIfNoFFmpeg(t)
	f, err := New("", "", nopLogger{})
	require.NoError(t, err)
	src, err := f.Probe("../../../testdata/short-sample.mp3")
	require.NoError(t, err)

	workDir := t.TempDir()
	// 1-second chunks with 200ms overlap.
	opts := ports.ChunkOpts{ChunkLengthSec: 1, OverlapSec: 0}
	chunksNoOverlap, err := f.Chunk(context.Background(), src, src.SizeBytes*10, workDir, opts)
	require.NoError(t, err)
	require.Greater(t, len(chunksNoOverlap), 1)

	workDir2 := t.TempDir()
	optsOverlap := ports.ChunkOpts{ChunkLengthSec: 1, OverlapSec: 1}
	chunksOverlap, err := f.Chunk(context.Background(), src, src.SizeBytes*10, workDir2, optsOverlap)
	require.NoError(t, err)
	require.Greater(t, len(chunksOverlap), 1)

	// Second chunk with overlap should start earlier (StartOffset = 0 because
	// nominal=1s, overlap=1s → max(0, 1s-1s)=0 — first chunk stays at 0).
	// Third chunk: nominal=2s, overlap=1s → StartOffset=1s instead of 2s.
	if len(chunksOverlap) >= 3 {
		require.Equal(t, time.Second, chunksOverlap[2].StartOffset,
			"third chunk should start 1s earlier than nominal 2s boundary")
	}
	// First chunk is always at 0 regardless of overlap.
	require.Equal(t, time.Duration(0), chunksOverlap[0].StartOffset)
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
