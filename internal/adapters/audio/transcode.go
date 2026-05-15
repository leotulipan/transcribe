package audio

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

// targetContainer returns the container/ext used to wrap a transcoded codec.
func targetContainer(codec string) (container, ext string, err error) {
	switch codec {
	case "mp3":
		return "mp3", ".mp3", nil
	case "flac":
		return "flac", ".flac", nil
	case "pcm_s16le":
		return "wav", ".wav", nil
	default:
		return "", "", fmt.Errorf("unsupported transcode codec %q", codec)
	}
}

// ffmpegCodecArgs returns the -c:a / -b:a / -ar / -ac args for a target.
func ffmpegCodecArgs(t ports.TargetFormat) []string {
	var args []string
	switch t.Codec {
	case "mp3":
		args = append(args, "-c:a", "libmp3lame")
	case "flac":
		args = append(args, "-c:a", "flac")
	case "pcm_s16le":
		args = append(args, "-c:a", "pcm_s16le")
	default:
		args = append(args, "-c:a", t.Codec)
	}
	if t.Bitrate != "" {
		args = append(args, "-b:a", t.Bitrate)
	}
	if t.SampleRate > 0 {
		args = append(args, "-ar", fmt.Sprintf("%d", t.SampleRate))
	}
	return args
}

func (f *FFmpeg) Transcode(ctx context.Context, in domain.AudioFile, t ports.TargetFormat, workDir string) (domain.AudioFile, error) {
	container, ext, err := targetContainer(t.Codec)
	if err != nil {
		return domain.AudioFile{}, err
	}
	if err := os.MkdirAll(workDir, 0o755); err != nil {
		return domain.AudioFile{}, err
	}
	base := strings.TrimSuffix(filepath.Base(in.Path), filepath.Ext(in.Path))
	final := filepath.Join(workDir, base+ext)
	partial := partialPath(final)

	args := []string{"-y", "-i", in.Path, "-vn"}
	args = append(args, ffmpegCodecArgs(t)...)
	// Specify format explicitly so ffmpeg doesn't get confused by .partial ext
	if fmtName, ok := containerFormat[container]; ok {
		args = append(args, "-f", fmtName)
	}
	args = append(args, partial)

	cmd := exec.CommandContext(ctx, f.ffmpeg, args...)
	if out, err := cmd.CombinedOutput(); err != nil {
		_ = os.Remove(partial)
		return domain.AudioFile{}, fmt.Errorf("ffmpeg transcode: %w: %s", err, string(out))
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
		Codec:     t.Codec,
		IsTemp:    true,
		Complete:  true,
	}, nil
}
