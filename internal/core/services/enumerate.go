package services

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// AudioExtensions is the canonical list of file extensions transcribe knows
// how to ingest. Update here to add format support everywhere at once.
var AudioExtensions = []string{
	".mp3", ".wav", ".m4a", ".flac", ".ogg", ".oga", ".opus",
	".mp4", ".mkv", ".mov", ".avi", ".webm",
}

// EnumerateAudioFiles returns the audio/video files at root. If root is a
// regular file it returns [root] (no extension filtering — the user picked it
// explicitly). If root is a directory it walks recursively, skips hidden
// files (leading "."), filters by AudioExtensions, and returns a sorted slice.
func EnumerateAudioFiles(root string) ([]string, error) {
	info, err := os.Stat(root)
	if err != nil {
		return nil, err
	}
	if !info.IsDir() {
		return []string{root}, nil
	}

	exts := make(map[string]struct{}, len(AudioExtensions))
	for _, e := range AudioExtensions {
		exts[e] = struct{}{}
	}

	var out []string
	err = filepath.WalkDir(root, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			// Skip unreadable dirs but keep walking.
			if errors.Is(walkErr, fs.ErrPermission) {
				return nil
			}
			return walkErr
		}
		name := d.Name()
		if d.IsDir() {
			// Don't descend into hidden directories.
			if name != "." && strings.HasPrefix(name, ".") {
				return fs.SkipDir
			}
			return nil
		}
		if strings.HasPrefix(name, ".") {
			return nil
		}
		ext := strings.ToLower(filepath.Ext(name))
		if _, ok := exts[ext]; !ok {
			return nil
		}
		out = append(out, path)
		return nil
	})
	if err != nil {
		return nil, err
	}
	sort.Strings(out)
	return out, nil
}
