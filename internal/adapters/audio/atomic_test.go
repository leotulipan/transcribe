package audio

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestPartialPath(t *testing.T) {
	require.Equal(t, "out.mp3.partial", partialPath("out.mp3"))
	require.Equal(t, filepath.Join("dir", "x.flac.partial"), partialPath(filepath.Join("dir", "x.flac")))
}

func TestPromote_RenamesAndReportsSize(t *testing.T) {
	dir := t.TempDir()
	final := filepath.Join(dir, "out.mp3")
	require.NoError(t, os.WriteFile(partialPath(final), []byte("12345"), 0o644))

	size, err := promote(final)
	require.NoError(t, err)
	require.Equal(t, int64(5), size)
	_, err = os.Stat(final)
	require.NoError(t, err)
	_, err = os.Stat(partialPath(final))
	require.ErrorIs(t, err, os.ErrNotExist)
}
