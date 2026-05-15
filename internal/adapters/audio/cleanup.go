package audio

import (
	"errors"
	"io/fs"
	"os"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

func (f *FFmpeg) Cleanup(file domain.AudioFile) error {
	if !file.IsTemp || file.Path == "" {
		return nil
	}
	var firstErr error
	for _, p := range []string{file.Path, metaPath(file.Path)} {
		if err := os.Remove(p); err != nil && !errors.Is(err, fs.ErrNotExist) && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}
