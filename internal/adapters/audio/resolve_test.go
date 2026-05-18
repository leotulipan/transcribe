package audio

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/stretchr/testify/require"
)

// makeFakeBin creates an empty file at dir/name (with .exe on Windows).
func makeFakeBin(t *testing.T, dir, name string) string {
	t.Helper()
	p := filepath.Join(dir, exeName(name))
	require.NoError(t, os.WriteFile(p, []byte{0x7f, 'E', 'L', 'F'}, 0o755))
	return p
}

func TestResolveBinary_UserProvidesFullFilePath(t *testing.T) {
	dir := t.TempDir()
	want := makeFakeBin(t, dir, "ffmpeg")

	got, err := ResolveBinary(want, "ffmpeg")
	require.NoError(t, err)
	gotAbs, _ := filepath.Abs(want)
	require.Equal(t, gotAbs, got)
}

func TestResolveBinary_UserProvidesDirectory(t *testing.T) {
	dir := t.TempDir()
	makeFakeBin(t, dir, "ffmpeg")

	got, err := ResolveBinary(dir, "ffmpeg")
	require.NoError(t, err)
	wantAbs, _ := filepath.Abs(filepath.Join(dir, exeName("ffmpeg")))
	require.Equal(t, wantAbs, got)
}

func TestResolveBinary_UserPathQuotedAndPadded(t *testing.T) {
	dir := t.TempDir()
	want := makeFakeBin(t, dir, "ffmpeg")
	// PowerShell users sometimes paste paths with surrounding quotes or trailing whitespace.
	padded := `  "` + dir + `"  `

	got, err := ResolveBinary(padded, "ffmpeg")
	require.NoError(t, err)
	wantAbs, _ := filepath.Abs(want)
	require.Equal(t, wantAbs, got)
}

func TestResolveBinary_UserProvidesMissingPath(t *testing.T) {
	bogus := filepath.Join(t.TempDir(), "does-not-exist")

	// Fall back to PATH walk. Inject a fake PATH dir with the binary.
	dir := t.TempDir()
	makeFakeBin(t, dir, "ffmpeg")
	t.Setenv("PATH", dir)

	got, err := ResolveBinary(bogus, "ffmpeg")
	require.NoError(t, err)
	require.Contains(t, got, "ffmpeg")
}

func TestResolveBinary_PATHWalkFallback(t *testing.T) {
	dir := t.TempDir()
	makeFakeBin(t, dir, "ffmpeg")

	// Empty PATH set to ONLY our temp dir. exec.LookPath will find it; this
	// also exercises the empty-user-path code path.
	t.Setenv("PATH", dir)

	got, err := ResolveBinary("", "ffmpeg")
	require.NoError(t, err)
	require.Contains(t, got, "ffmpeg")
}

func TestResolveBinary_NotFoundAnywhere(t *testing.T) {
	t.Setenv("PATH", t.TempDir()) // empty dir, no binary
	if runtime.GOOS == "windows" {
		// Also blank out the WinGet/shim env vars so the shim search misses.
		t.Setenv("LOCALAPPDATA", t.TempDir())
		t.Setenv("USERPROFILE", t.TempDir())
	}

	_, err := ResolveBinary("", "definitely-not-a-real-binary-name-xyz")
	require.Error(t, err)
	require.ErrorIs(t, err, ErrBinaryNotFound)
}

func TestResolveBinary_RejectsUserPathThatIsActuallyADirOnly(t *testing.T) {
	// User supplied a directory but the binary isn't in it; expect we fall
	// through to PATH search rather than returning success.
	emptyDir := t.TempDir()
	pathDir := t.TempDir()
	makeFakeBin(t, pathDir, "ffmpeg")
	t.Setenv("PATH", pathDir)
	if runtime.GOOS == "windows" {
		// Isolate from any real WinGet/Chocolatey/Scoop installs on the host.
		t.Setenv("LOCALAPPDATA", t.TempDir())
		t.Setenv("USERPROFILE", t.TempDir())
	}

	got, err := ResolveBinary(emptyDir, "ffmpeg")
	require.NoError(t, err)
	gotAbs, _ := filepath.Abs(got)
	wantAbs, _ := filepath.Abs(filepath.Join(pathDir, exeName("ffmpeg")))
	require.Equal(t, wantAbs, gotAbs)
}

func TestResolveBinary_ErrorListsAttemptedPaths(t *testing.T) {
	bogusUser := filepath.Join(t.TempDir(), "nope")
	t.Setenv("PATH", t.TempDir())
	if runtime.GOOS == "windows" {
		t.Setenv("LOCALAPPDATA", t.TempDir())
		t.Setenv("USERPROFILE", t.TempDir())
	}

	_, err := ResolveBinary(bogusUser, "ffmpeg-xyz-missing")
	require.Error(t, err)
	require.Contains(t, err.Error(), "ffmpeg-xyz-missing")
	require.True(t, errors.Is(err, ErrBinaryNotFound))
}
