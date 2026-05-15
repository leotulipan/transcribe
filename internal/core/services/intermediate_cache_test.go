package services

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/adapters/audio"
	"github.com/leotulipan/transcribe/internal/core/domain"
)

func TestLookupIntermediate_MatchesAndIgnoresMismatch(t *testing.T) {
	dir := t.TempDir()
	inter := filepath.Join(dir, "talk.m4a")
	require.NoError(t, os.WriteFile(inter, []byte("data"), 0o644))

	src := domain.AudioFile{Path: "/tmp/talk.mp4", Codec: "aac", Container: "mp4", SizeBytes: 1000}
	require.NoError(t, audio.WriteMeta(inter, audio.MetaInfo{
		Operation: "copy", SourcePath: src.Path, SourceSize: src.SizeBytes,
		SourceMTimeUnix: 0, TargetCodec: "aac", TargetContainer: "m4a",
		MaxBytesBudget: 25 << 20, Provider: domain.ProviderGroq, Model: "whisper-large-v3",
	}))

	// matching lookup
	got := lookupIntermediate(dir, src, 0, domain.ProviderGroq, "whisper-large-v3", 25<<20, "aac")
	require.NotNil(t, got)
	require.Equal(t, inter, got.Path)

	// mismatching codec → nil
	none := lookupIntermediate(dir, src, 0, domain.ProviderGroq, "whisper-large-v3", 25<<20, "mp3")
	require.Nil(t, none)
}
