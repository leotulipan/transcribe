package ports

import (
    "context"

    "github.com/leotulipan/transcribe/internal/core/domain"
)

// AudioProcessor wraps ffmpeg/ffprobe operations.
type AudioProcessor interface {
    Probe(path string) (domain.AudioFile, error)

    // CopyAudio stream-copies the audio track into a derived container without
    // re-encoding. workDir is the directory in which to land the output.
    CopyAudio(ctx context.Context, in domain.AudioFile, workDir string) (domain.AudioFile, error)

    // ExtractAudio decodes to 16-bit mono PCM WAV — the lossless fallback when
    // stream copy isn't viable.
    ExtractAudio(ctx context.Context, videoPath string, workDir string) (domain.AudioFile, error)

    // Transcode re-encodes to a target codec/bitrate.
    Transcode(ctx context.Context, in domain.AudioFile, target TargetFormat, workDir string) (domain.AudioFile, error)

    // Chunk slices a file into byte-size-bounded chunks.
    Chunk(ctx context.Context, in domain.AudioFile, maxBytes int64, workDir string) ([]domain.Chunk, error)

    // Cleanup removes one tempfile. Caller decides when based on the cleanup
    // policy in services.
    Cleanup(f domain.AudioFile) error
}

type TargetFormat struct {
    Codec      string // "flac" | "mp3" | "pcm_s16le"
    Bitrate    string // empty for flac/pcm
    SampleRate int    // 0 = keep source
}
