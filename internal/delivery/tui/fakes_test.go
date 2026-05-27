package tui

import (
	"context"

	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

// fakeSvc is a minimal in-memory TranscribeService for TUI tests.
type fakeSvc struct{}

func (s *fakeSvc) Submit(_ context.Context, req domain.Request) (ports.Job, error) {
	return &fakeJob{}, nil
}

func (s *fakeSvc) ListProviders() []domain.ProviderID {
	return []domain.ProviderID{domain.ProviderGroq, domain.ProviderOpenAI}
}

func (s *fakeSvc) DefaultModel(_ domain.ProviderID) string { return "whisper-large-v3" }

func (s *fakeSvc) ListModels(p domain.ProviderID) ([]string, error) {
	return []string{"whisper-large-v3", "whisper-large-v3-turbo"}, nil
}

func (s *fakeSvc) DiscoverModels(_ context.Context, p domain.ProviderID) ([]string, error) {
	return []string{"whisper-large-v3", "whisper-large-v3-turbo"}, nil
}

// fakeJob completes immediately with a synthetic result.
type fakeJob struct {
	done chan struct{}
	once bool
}

func (j *fakeJob) ID() string { return "fake-job-id" }

func (j *fakeJob) Progress() <-chan domain.ProgressEvent {
	ch := make(chan domain.ProgressEvent, 2)
	ch <- domain.ProgressEvent{Stage: domain.StageProbing, Message: "probing", Percent: 0.1}
	ch <- domain.ProgressEvent{Stage: domain.StageDone, Message: "done", Percent: 1.0}
	close(ch)
	return ch
}

func (j *fakeJob) Wait() (*domain.Result, error) {
	return &domain.Result{
		Provider: domain.ProviderGroq,
		Text:     "Hello, world. This is a test transcription.",
	}, nil
}

func (j *fakeJob) Cancel() {}
