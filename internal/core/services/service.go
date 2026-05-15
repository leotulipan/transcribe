package services

import (
	"context"
	"time"

	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

// Deps bundles the output ports the service needs.
type Deps struct {
	Providers map[domain.ProviderID]ports.Provider
	Audio     ports.AudioProcessor
	Cache     ports.ResultCache
	Writers   map[domain.OutputFormat]ports.FormatWriter
	Log       ports.Logger
}

// Service implements ports.TranscribeService.
type Service struct {
	deps Deps
}

func New(deps Deps) *Service { return &Service{deps: deps} }

var _ ports.TranscribeService = (*Service)(nil)

func (s *Service) ListProviders() []domain.ProviderID {
	out := make([]domain.ProviderID, 0, len(s.deps.Providers))
	for k := range s.deps.Providers {
		out = append(out, k)
	}
	return out
}

func (s *Service) ListModels(id domain.ProviderID) ([]string, error) {
	p, err := providerFor(s.deps, id)
	if err != nil {
		return nil, err
	}
	return p.Models(), nil
}

// Submit is implemented in job.go (Task K2).
func (s *Service) Submit(ctx context.Context, req domain.Request) (ports.Job, error) {
	return s.submit(ctx, req)
}

func (s *Service) submit(parent context.Context, req domain.Request) (ports.Job, error) {
	j := newJob(parent, req, generateJobID())
	go j.run(func(emit func(domain.ProgressEvent)) (*domain.Result, error) {
		return pipelineRun(j.ctx, req, s.deps, emit)
	})
	return j, nil
}

func generateJobID() string {
	// crypto/rand-backed in real life; for v1 a timestamp is fine
	return time.Now().UTC().Format("20060102T150405.000000000")
}
