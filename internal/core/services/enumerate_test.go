package services

import (
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/stretchr/testify/require"
)

func touch(t *testing.T, path string) {
	t.Helper()
	require.NoError(t, os.MkdirAll(filepath.Dir(path), 0o755))
	require.NoError(t, os.WriteFile(path, []byte("x"), 0o644))
}

func TestEnumerateAudioFiles_SingleFileReturnedAsIs(t *testing.T) {
	dir := t.TempDir()
	f := filepath.Join(dir, "song.mp3")
	touch(t, f)

	got, err := EnumerateAudioFiles(f)
	require.NoError(t, err)
	require.Equal(t, []string{f}, got)
}

func TestEnumerateAudioFiles_SingleFileNoExtFilter(t *testing.T) {
	// When the user explicitly picks a file we shouldn't filter by extension.
	dir := t.TempDir()
	f := filepath.Join(dir, "weird.bin")
	touch(t, f)

	got, err := EnumerateAudioFiles(f)
	require.NoError(t, err)
	require.Equal(t, []string{f}, got)
}

func TestEnumerateAudioFiles_DirRecursiveFiltered(t *testing.T) {
	dir := t.TempDir()
	touch(t, filepath.Join(dir, "a.mp3"))
	touch(t, filepath.Join(dir, "b.wav"))
	touch(t, filepath.Join(dir, "ignore.txt"))
	touch(t, filepath.Join(dir, "sub", "c.flac"))
	touch(t, filepath.Join(dir, "sub", "deep", "d.mov"))
	touch(t, filepath.Join(dir, "sub", "notes.md"))

	got, err := EnumerateAudioFiles(dir)
	require.NoError(t, err)

	want := []string{
		filepath.Join(dir, "a.mp3"),
		filepath.Join(dir, "b.wav"),
		filepath.Join(dir, "sub", "c.flac"),
		filepath.Join(dir, "sub", "deep", "d.mov"),
	}
	sort.Strings(want)
	require.Equal(t, want, got)
}

func TestEnumerateAudioFiles_SkipsHiddenFilesAndDirs(t *testing.T) {
	dir := t.TempDir()
	touch(t, filepath.Join(dir, ".hidden.mp3"))
	touch(t, filepath.Join(dir, ".secrets", "leak.wav"))
	touch(t, filepath.Join(dir, "visible.mp3"))

	got, err := EnumerateAudioFiles(dir)
	require.NoError(t, err)
	require.Equal(t, []string{filepath.Join(dir, "visible.mp3")}, got)
}

func TestEnumerateAudioFiles_CaseInsensitiveExt(t *testing.T) {
	dir := t.TempDir()
	touch(t, filepath.Join(dir, "A.MP3"))
	touch(t, filepath.Join(dir, "b.Wav"))

	got, err := EnumerateAudioFiles(dir)
	require.NoError(t, err)
	require.Len(t, got, 2)
}

func TestEnumerateAudioFiles_EmptyDir(t *testing.T) {
	dir := t.TempDir()
	got, err := EnumerateAudioFiles(dir)
	require.NoError(t, err)
	require.Empty(t, got)
}

func TestEnumerateAudioFiles_MissingPath(t *testing.T) {
	_, err := EnumerateAudioFiles(filepath.Join(t.TempDir(), "does-not-exist"))
	require.Error(t, err)
}

func TestEnumerateAudioFilesWith_ExtensionsOverridesDefault(t *testing.T) {
	dir := t.TempDir()
	touch(t, filepath.Join(dir, "a.mp3"))
	touch(t, filepath.Join(dir, "b.wav"))
	touch(t, filepath.Join(dir, "c.m4a"))

	// Only mp3 and m4a — wav must be excluded.
	got, err := EnumerateAudioFilesWith(dir, []string{"mp3", "m4a"})
	require.NoError(t, err)

	want := []string{
		filepath.Join(dir, "a.mp3"),
		filepath.Join(dir, "c.m4a"),
	}
	sort.Strings(want)
	require.Equal(t, want, got)
}

func TestEnumerateAudioFilesWith_DotPrefixAccepted(t *testing.T) {
	dir := t.TempDir()
	touch(t, filepath.Join(dir, "a.mp3"))
	touch(t, filepath.Join(dir, "b.wav"))

	// ".mp3" with leading dot must also work.
	got, err := EnumerateAudioFilesWith(dir, []string{".mp3"})
	require.NoError(t, err)
	require.Equal(t, []string{filepath.Join(dir, "a.mp3")}, got)
}

func TestEnumerateAudioFilesWith_NoExtensionsUsesDefault(t *testing.T) {
	dir := t.TempDir()
	touch(t, filepath.Join(dir, "a.mp3"))
	touch(t, filepath.Join(dir, "ignore.txt"))

	got, err := EnumerateAudioFilesWith(dir, nil)
	require.NoError(t, err)
	require.Equal(t, []string{filepath.Join(dir, "a.mp3")}, got)
}
