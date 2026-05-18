package audio

import (
	"errors"

	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

// FFmpeg implements ports.AudioProcessor.
type FFmpeg struct {
	ffmpeg  string
	ffprobe string
	log     ports.Logger
}

// New resolves ffmpeg and ffprobe via ResolveBinary, falling back to PATH
// discovery if the user-supplied paths are empty or invalid. Returns
// domain.ErrFFmpegMissing wrapping the resolution failure when either binary
// cannot be located.
func New(ffmpegPath, ffprobePath string, log ports.Logger) (*FFmpeg, error) {
	resolvedFFmpeg, err := ResolveBinary(ffmpegPath, "ffmpeg")
	if err != nil {
		// If a user-supplied path was invalid, retry with auto-discover so the
		// app still works when ffmpeg is on PATH. Warn so the bad value gets
		// surfaced. Without a logger, just swallow it.
		if ffmpegPath != "" {
			if log != nil {
				log.Warn("configured ffmpeg path invalid; falling back to PATH",
					"path", ffmpegPath, "err", err.Error())
			}
			if p2, err2 := ResolveBinary("", "ffmpeg"); err2 == nil {
				resolvedFFmpeg = p2
			} else {
				return nil, errors.Join(domain.ErrFFmpegMissing, err2)
			}
		} else {
			return nil, errors.Join(domain.ErrFFmpegMissing, err)
		}
	}
	resolvedFFprobe, err := ResolveBinary(ffprobePath, "ffprobe")
	if err != nil {
		if ffprobePath != "" {
			if log != nil {
				log.Warn("configured ffprobe path invalid; falling back to PATH",
					"path", ffprobePath, "err", err.Error())
			}
			if p2, err2 := ResolveBinary("", "ffprobe"); err2 == nil {
				resolvedFFprobe = p2
			} else {
				return nil, errors.Join(domain.ErrFFmpegMissing, err2)
			}
		} else {
			return nil, errors.Join(domain.ErrFFmpegMissing, err)
		}
	}
	return &FFmpeg{ffmpeg: resolvedFFmpeg, ffprobe: resolvedFFprobe, log: log}, nil
}

// compile-time check
var _ ports.AudioProcessor = (*FFmpeg)(nil)
