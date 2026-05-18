package audio

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

func (f *FFmpeg) ExtractAudio(ctx context.Context, videoPath, workDir string) (domain.AudioFile, error) {
	if err := os.MkdirAll(workDir, 0o755); err != nil {
		return domain.AudioFile{}, err
	}
	base := strings.TrimSuffix(filepath.Base(videoPath), filepath.Ext(videoPath))
	final := filepath.Join(workDir, base+".wav")
	partial := partialPath(final)

	cmd := exec.CommandContext(ctx, f.ffmpeg,
		"-y", "-i", videoPath, "-vn",
		"-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
		"-f", "wav",
		partial,
	)
	if out, err := cmd.CombinedOutput(); err != nil {
		_ = os.Remove(partial)
		return domain.AudioFile{}, fmt.Errorf("ffmpeg extract: %w: %s", err, string(out))
	}
	size, err := promote(final)
	if err != nil {
		return domain.AudioFile{}, err
	}

	af, err := f.Probe(final)
	if err != nil {
		return domain.AudioFile{}, err
	}
	af.IsTemp = true
	af.Complete = true
	af.SizeBytes = size
	return af, nil
}
