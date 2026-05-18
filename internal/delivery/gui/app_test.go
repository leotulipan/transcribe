package gui

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

// fakeService is a minimal ports.TranscribeService for Reload tests.
type fakeService struct{ id string }

func (f *fakeService) ListProviders() []domain.ProviderID { return nil }
func (f *fakeService) ListModels(domain.ProviderID) ([]string, error) {
	return nil, nil
}
func (f *fakeService) DiscoverModels(context.Context, domain.ProviderID) ([]string, error) {
	return nil, nil
}
func (f *fakeService) Submit(context.Context, domain.Request) (ports.Job, error) {
	return nil, errors.New("not implemented")
}

func TestDeps_ReloadSwapsServiceAndConfig(t *testing.T) {
	initialCfg := ports.Config{APIKeys: map[domain.ProviderID]string{
		domain.ProviderGroq: "v1",
	}}
	updatedCfg := ports.Config{APIKeys: map[domain.ProviderID]string{
		domain.ProviderGroq: "v2",
	}}

	svc1 := &fakeService{id: "v1"}
	svc2 := &fakeService{id: "v2"}

	loadCalls := atomic.Int32{}
	d := NewDeps(svc1, initialCfg, nil, nil,
		func() (ports.Config, error) {
			loadCalls.Add(1)
			return updatedCfg, nil
		},
		func(c ports.Config) (ports.TranscribeService, error) {
			require.Equal(t, "v2", c.APIKeys[domain.ProviderGroq])
			return svc2, nil
		},
	)

	// Before reload, the original service and config are visible.
	require.Same(t, svc1, d.Service())
	require.Equal(t, "v1", d.Config().APIKeys[domain.ProviderGroq])

	require.NoError(t, d.Reload())

	require.Same(t, svc2, d.Service())
	require.Equal(t, "v2", d.Config().APIKeys[domain.ProviderGroq])
	require.Equal(t, int32(1), loadCalls.Load())
}

func TestDeps_ReloadErrorsWhenNotWired(t *testing.T) {
	d := NewDeps(&fakeService{}, ports.Config{}, nil, nil, nil, nil)
	err := d.Reload()
	require.ErrorIs(t, err, ErrReloadNotWired)
}

func TestDeps_ReloadPreservesOldServiceForInFlightCallers(t *testing.T) {
	// Simulate: a caller captures Service() before Reload, holds it past
	// the reload. The captured pointer must still be valid afterwards.
	svc1 := &fakeService{id: "v1"}
	svc2 := &fakeService{id: "v2"}
	d := NewDeps(svc1, ports.Config{}, nil, nil,
		func() (ports.Config, error) { return ports.Config{}, nil },
		func(ports.Config) (ports.TranscribeService, error) { return svc2, nil },
	)

	captured := d.Service()
	require.NoError(t, d.Reload())

	// captured (svc1) is unchanged; Service() now returns svc2.
	require.Same(t, svc1, captured)
	require.Same(t, svc2, d.Service())
}

func TestDeps_ConcurrentReadsAndReloads(t *testing.T) {
	// Hammer the RWMutex from many goroutines to surface any race.
	calls := atomic.Int32{}
	d := NewDeps(&fakeService{id: "0"}, ports.Config{}, nil, nil,
		func() (ports.Config, error) { return ports.Config{}, nil },
		func(ports.Config) (ports.TranscribeService, error) {
			n := calls.Add(1)
			return &fakeService{id: string(rune('a' + n%26))}, nil
		},
	)

	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = d.Service()
			_ = d.Config()
		}()
	}
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = d.Reload()
		}()
	}
	wg.Wait()
	require.Greater(t, calls.Load(), int32(0))
}
