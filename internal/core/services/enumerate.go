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

// EnumerateAudioFiles returns the audio/video files at root using the default
// AudioExtensions list. See EnumerateAudioFilesWith for the full contract.
func EnumerateAudioFiles(root string) ([]string, error) {
	return EnumerateAudioFilesWith(root, nil)
}

// EnumerateAudioFilesWith returns the audio/video files at root. If root is a
// regular file it returns [root] (no extension filtering — the user picked it
// explicitly). If root is a directory it walks recursively, skips hidden
// files (leading "."), and filters by the provided extensions list.
//
// extensions may be nil or empty, in which case AudioExtensions is used. Each
// entry is normalised: a leading dot is added when absent and the value is
// lowercased. Both "mp3" and ".mp3" are accepted.
func EnumerateAudioFilesWith(root string, extensions []string) ([]string, error) {
	info, err := os.Stat(root)
	if err != nil {
		return nil, err
	}
	if !info.IsDir() {
		return []string{root}, nil
	}

	list := AudioExtensions
	if len(extensions) > 0 {
		list = extensions
	}
	exts := make(map[string]struct{}, len(list))
	for _, e := range list {
		e = strings.ToLower(e)
		if !strings.HasPrefix(e, ".") {
			e = "." + e
		}
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
