package services

import (
	"context"
	"errors"
	"time"

	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

// ErrDiscoveryUnsupported is returned by DiscoverModels when the provider
// adapter does not implement ports.ModelDiscoverer.
var ErrDiscoveryUnsupported = errors.New("provider does not support live model discovery")

// Deps bundles the output ports the service needs.
type Deps struct {
	Providers map[domain.ProviderID]ports.Provider
	Audio     ports.AudioProcessor
	Cache     ports.ResultCache
	Writers   map[domain.OutputFormat]ports.FormatWriter
	Log       ports.Logger
	// DiscoveredModels overrides each provider's hardcoded Models() with a
	// live list previously fetched via DiscoverModels. Nil/missing entries
	// fall back to the adapter's hardcoded slice.
	DiscoveredModels map[domain.ProviderID][]string
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
	// Discovered list replaces the hardcoded fallback. Per-plan decision:
	// trust the live result over the bundled defaults.
	if list, ok := s.deps.DiscoveredModels[id]; ok && len(list) > 0 {
		return list, nil
	}
	return p.Models(), nil
}

// DiscoverModels invokes the provider's live model-listing endpoint, if it
// implements ports.ModelDiscoverer. Returns (nil, ports.ErrUnsupportedCapability)
// for adapters that don't support discovery.
func (s *Service) DiscoverModels(ctx context.Context, id domain.ProviderID) ([]string, error) {
	p, err := providerFor(s.deps, id)
	if err != nil {
		return nil, err
	}
	disc, ok := p.(ports.ModelDiscoverer)
	if !ok {
		return nil, ErrDiscoveryUnsupported
	}
	return disc.DiscoverModels(ctx)
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
