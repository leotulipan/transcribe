package ports

import (
    "context"

    "github.com/leotulipan/transcribe/internal/core/domain"
)

// Provider is implemented by each transcription API adapter.
type Provider interface {
    ID() domain.ProviderID
    MaxUploadBytes() int64
    Models() []string
    DefaultModel() string

    // Capabilities returns model-level capabilities. Service validates them
    // against Request.Formats before any audio work begins.
    Capabilities(model string) ModelCapabilities

    // Transcribe ingests a file already within MaxUploadBytes() and an accepted
    // codec, returns a normalized Result.
    Transcribe(ctx context.Context, audio domain.AudioFile, opts ProviderOpts) (*domain.Result, error)

    // CheckKey performs a non-consuming probe (e.g. GET /models) to verify the
    // configured API key is valid. Returns nil on success. Adapters that do
    // not yet implement a probe return errors.ErrUnsupported.
    CheckKey(ctx context.Context) error
}

type ModelCapabilities struct {
    WordTimestamps    bool
    SegmentTimestamps bool
    Diarization       bool                  // informational in v1
    LanguageHint      bool
    AcceptedInputs    []domain.AudioFormat
}

type ProviderOpts struct {
    Model         string
    Language      string
    SpeakerLabels bool // request diarization; providers that don't support it silently ignore this
}
