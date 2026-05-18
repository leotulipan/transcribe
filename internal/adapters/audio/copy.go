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

// codecContainer maps an audio codec to the container we will wrap it in.
var codecContainer = map[string]string{
	"aac":       "m4a",
	"alac":      "m4a",
	"mp3":       "mp3",
	"opus":      "ogg",
	"vorbis":    "ogg",
	"flac":      "flac",
	"pcm_s16le": "wav",
	"pcm_s24le": "wav",
	"pcm_f32le": "wav",
}

// containerExt returns the file extension for a copy-target container.
func containerExt(container string) string {
	return "." + container
}

// containerFormat maps a container name to the ffmpeg format name for -f.
var containerFormat = map[string]string{
	"m4a":  "ipod",
	"mp3":  "mp3",
	"ogg":  "ogg",
	"flac": "flac",
	"wav":  "wav",
}

func (f *FFmpeg) CopyAudio(ctx context.Context, in domain.AudioFile, workDir string) (domain.AudioFile, error) {
	container, ok := codecContainer[in.Codec]
	if !ok {
		return domain.AudioFile{}, fmt.Errorf("copy-audio: codec %q has no known container", in.Codec)
	}
	if err := os.MkdirAll(workDir, 0o755); err != nil {
		return domain.AudioFile{}, err
	}
	base := strings.TrimSuffix(filepath.Base(in.Path), filepath.Ext(in.Path))
	final := filepath.Join(workDir, base+containerExt(container))
	partial := partialPath(final)

	fmtArgs := []string{}
	if fmtName, ok := containerFormat[container]; ok {
		fmtArgs = []string{"-f", fmtName}
	}
	args := append([]string{"-y", "-i", in.Path, "-vn", "-c:a", "copy"}, fmtArgs...)
	args = append(args, partial)
	cmd := exec.CommandContext(ctx, f.ffmpeg, args...)
	if out, err := cmd.CombinedOutput(); err != nil {
		_ = os.Remove(partial)
		return domain.AudioFile{}, fmt.Errorf("ffmpeg copy: %w: %s", err, string(out))
	}
	size, err := promote(final)
	if err != nil {
		return domain.AudioFile{}, err
	}

	return domain.AudioFile{
		Path:      final,
		SizeBytes: size,
		Duration:  in.Duration,
		Container: container,
		Codec:     in.Codec,
		IsTemp:    true,
		Complete:  true,
	}, nil
}
