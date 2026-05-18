package audio

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

// ErrBinaryNotFound is returned by ResolveBinary when no path could be
// resolved to a regular executable file. Use errors.Is to test for it.
var ErrBinaryNotFound = errors.New("binary not found")

// ResolveBinary returns an absolute path to the named binary (e.g. "ffmpeg",
// "ffprobe") or wraps ErrBinaryNotFound with a list of locations attempted.
//
// Resolution order (first hit wins):
//  1. userPath, if non-empty. A directory is upgraded to dir/binName(.exe).
//  2. Well-known package-manager shim locations (Windows: WinGet Links,
//     Chocolatey bin, Scoop shims).
//  3. exec.LookPath — the stdlib PATH walk (honors PATHEXT on Windows).
//  4. Explicit PATH walk — catches edge cases LookPath misses (quoted entries,
//     trailing whitespace, etc.).
//
// The returned path is always validated with os.Stat to be a regular file.
func ResolveBinary(userPath, binName string) (string, error) {
	var tried []string

	// 1. User-provided path.
	if userPath != "" {
		p, ok := normalize(userPath, binName)
		tried = append(tried, p)
		if ok {
			abs, _ := filepath.Abs(p)
			return abs, nil
		}
	}

	// 2. Well-known shim locations (Windows only).
	for _, dir := range knownShimDirs() {
		candidate := filepath.Join(dir, exeName(binName))
		tried = append(tried, candidate)
		if isRegularFile(candidate) {
			abs, _ := filepath.Abs(candidate)
			return abs, nil
		}
	}

	// 3. exec.LookPath.
	if p, err := exec.LookPath(binName); err == nil {
		abs, _ := filepath.Abs(p)
		return abs, nil
	}
	tried = append(tried, "PATH:"+binName)

	// 4. Explicit PATH walk.
	pathEnv := os.Getenv("PATH")
	for _, dir := range filepath.SplitList(pathEnv) {
		dir = strings.TrimSpace(strings.Trim(dir, `"`))
		if dir == "" {
			continue
		}
		candidate := filepath.Join(dir, exeName(binName))
		if isRegularFile(candidate) {
			abs, _ := filepath.Abs(candidate)
			return abs, nil
		}
	}

	return "", fmt.Errorf("%w: %s (tried: %s)",
		ErrBinaryNotFound, binName, strings.Join(tried, ", "))
}

// normalize takes a user-supplied path and a binary name. If the path is a
// directory, it appends the platform-correct binary filename. Returns
// (resolved-path, ok) where ok is true iff the resolved path is a regular file.
func normalize(userPath, binName string) (string, bool) {
	p := strings.TrimSpace(userPath)
	p = strings.Trim(p, `"`)
	if p == "" {
		return p, false
	}
	info, err := os.Stat(p)
	if err == nil && info.IsDir() {
		p = filepath.Join(p, exeName(binName))
	}
	return p, isRegularFile(p)
}

func isRegularFile(p string) bool {
	info, err := os.Stat(p)
	return err == nil && info.Mode().IsRegular()
}

func exeName(binName string) string {
	if runtime.GOOS == "windows" && !strings.HasSuffix(strings.ToLower(binName), ".exe") {
		return binName + ".exe"
	}
	return binName
}

// knownShimDirs returns directories where package managers commonly drop
// shim scripts on Windows. Empty on non-Windows.
func knownShimDirs() []string {
	if runtime.GOOS != "windows" {
		return nil
	}
	var dirs []string
	if v := os.Getenv("LOCALAPPDATA"); v != "" {
		dirs = append(dirs, filepath.Join(v, "Microsoft", "WinGet", "Links"))
	}
	dirs = append(dirs, `C:\ProgramData\chocolatey\bin`)
	if v := os.Getenv("USERPROFILE"); v != "" {
		dirs = append(dirs, filepath.Join(v, "scoop", "shims"))
	}
	return dirs
}
