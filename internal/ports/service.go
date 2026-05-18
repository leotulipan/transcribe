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
