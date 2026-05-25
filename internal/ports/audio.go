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

    // Chunk slices a file into byte-size-bounded chunks. opts tunes chunk
    // duration and overlap; zero values preserve current derived behaviour.
    Chunk(ctx context.Context, in domain.AudioFile, maxBytes int64, workDir string, opts ChunkOpts) ([]domain.Chunk, error)

    // Cleanup removes one tempfile. Caller decides when based on the cleanup
    // policy in services.
    Cleanup(f domain.AudioFile) error
}

type TargetFormat struct {
    Codec      string // "flac" | "mp3" | "pcm_s16le"
    Bitrate    string // empty for flac/pcm
    SampleRate int    // 0 = keep source
}

// PrepareOpts groups optional user-controlled knobs for the prepare step.
type PrepareOpts struct {
    // UseInput bypasses all conversion — the source is returned as-is regardless
    // of codec, container, or size.
    UseInput bool

    // SizeThresholdBytes widens the as-is path: when > 0 and the source is
    // codec-compatible, the file is returned as-is even when its size exceeds the
    // provider's maxBytes limit (up to this threshold). The user accepts that the
    // provider may then reject the upload.
    SizeThresholdBytes int64
}

// ChunkOpts groups optional user-controlled knobs for the chunker.
type ChunkOpts struct {
    // ChunkLengthSec fixes the chunk duration in seconds. When 0, the chunker
    // derives the duration from the byte budget and source bitrate.
    ChunkLengthSec int

    // OverlapSec makes each chunk (except the first) start this many seconds
    // before the nominal boundary. Words in the overlapping region are
    // double-transcribed; mergeChunks does NOT deduplicate them.
    OverlapSec int
}
