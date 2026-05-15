package audio

import "os"

// partialPath returns the in-flight name used while ffmpeg is still writing.
func partialPath(finalPath string) string {
	return finalPath + ".partial"
}

// promote renames "<finalPath>.partial" to "<finalPath>" once ffmpeg exits 0.
// Returns the file size of the promoted file.
func promote(finalPath string) (int64, error) {
	if err := os.Rename(partialPath(finalPath), finalPath); err != nil {
		return 0, err
	}
	info, err := os.Stat(finalPath)
	if err != nil {
		return 0, err
	}
	return info.Size(), nil
}
