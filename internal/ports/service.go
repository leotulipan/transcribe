package ports

import (
    "context"

    "github.com/leotulipan/transcribe/internal/core/domain"
)

// TranscribeService is the input port the delivery layer calls.
type TranscribeService interface {
    // Submit kicks off a transcription on a background goroutine and returns a
    // Job handle. The service owns the goroutine; the caller owns the handle.
    Submit(ctx context.Context, req domain.Request) (Job, error)

    // ListProviders returns providers configured at startup (API key present).
    ListProviders() []domain.ProviderID

    // ListModels returns the model names a provider advertises (preferring
    // discovered lists from the config when present).
    ListModels(p domain.ProviderID) ([]string, error)

    // DefaultModel returns the canonical default model for a provider — the
    // one UIs should pre-select. Returns "" if the provider is unknown.
    DefaultModel(p domain.ProviderID) string

    // Capabilities reports what a provider/model supports (word timestamps,
    // diarization, …) so UIs can hide or disable options that won't work.
    // ok is false when the provider isn't configured.
    Capabilities(p domain.ProviderID, model string) (ModelCapabilities, bool)

    // DiscoverModels invokes the provider's live "list models" endpoint.
    // Returns an error if the provider doesn't support live discovery.
    DiscoverModels(ctx context.Context, p domain.ProviderID) ([]string, error)
}

// Job is a handle to an in-flight (or finished) transcription.
type Job interface {
    ID() string
    Progress() <-chan domain.ProgressEvent // closed when the job ends
    Wait() (*domain.Result, error)         // blocks until done; safe for repeated calls
    Cancel()                                // idempotent
}
