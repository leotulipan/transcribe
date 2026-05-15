package audio

import (
	"context"
	"errors"
	"os/exec"

	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

// FFmpeg implements ports.AudioProcessor.
type FFmpeg struct {
	ffmpeg  string
	ffprobe string
	log     ports.Logger
}

func New(ffmpegPath, ffprobePath string, log ports.Logger) (*FFmpeg, error) {
	if ffmpegPath == "" {
		p, err := exec.LookPath("ffmpeg")
		if err != nil {
			return nil, domain.ErrFFmpegMissing
		}
		ffmpegPath = p
	}
	if ffprobePath == "" {
		p, err := exec.LookPath("ffprobe")
		if err != nil {
			return nil, domain.ErrFFmpegMissing
		}
		ffprobePath = p
	}
	return &FFmpeg{ffmpeg: ffmpegPath, ffprobe: ffprobePath, log: log}, nil
}

// compile-time check
var _ ports.AudioProcessor = (*FFmpeg)(nil)

// errInternal is a placeholder so unimplemented methods compile in early tasks.
var errInternal = errors.New("not implemented")

func (f *FFmpeg) ExtractAudio(ctx context.Context, videoPath, workDir string) (domain.AudioFile, error) {
	return domain.AudioFile{}, errInternal
}
func (f *FFmpeg) Transcode(ctx context.Context, in domain.AudioFile, t ports.TargetFormat, workDir string) (domain.AudioFile, error) {
	return domain.AudioFile{}, errInternal
}
func (f *FFmpeg) Chunk(ctx context.Context, in domain.AudioFile, maxBytes int64, workDir string) ([]domain.Chunk, error) {
	return nil, errInternal
}
func (f *FFmpeg) Cleanup(file domain.AudioFile) error { return errInternal }
