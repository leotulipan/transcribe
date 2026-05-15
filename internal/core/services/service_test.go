package services

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

type fakeProvider struct {
	id     domain.ProviderID
	models []string
}

func (f fakeProvider) ID() domain.ProviderID    { return f.id }
func (fakeProvider) MaxUploadBytes() int64      { return 25 << 20 }
func (f fakeProvider) Models() []string         { return f.models }
func (f fakeProvider) DefaultModel() string {
	if len(f.models) == 0 {
		return ""
	}
	return f.models[0]
}
func (fakeProvider) Capabilities(string) ports.ModelCapabilities { return ports.ModelCapabilities{} }
func (fakeProvider) Transcribe(_ context.Context, _ domain.AudioFile, _ ports.ProviderOpts) (*domain.Result, error) {
	panic("not called in this test")
}

func TestService_ListProviders_ReturnsConfigured(t *testing.T) {
	svc := New(Deps{
		Providers: map[domain.ProviderID]ports.Provider{
			domain.ProviderGroq:   fakeProvider{id: domain.ProviderGroq, models: []string{"whisper-large-v3"}},
			domain.ProviderOpenAI: fakeProvider{id: domain.ProviderOpenAI, models: []string{"whisper-1"}},
		},
	})
	got := svc.ListProviders()
	require.ElementsMatch(t, []domain.ProviderID{domain.ProviderGroq, domain.ProviderOpenAI}, got)
}

func TestService_ListModels_ReturnsProviderModels(t *testing.T) {
	svc := New(Deps{
		Providers: map[domain.ProviderID]ports.Provider{
			domain.ProviderGroq: fakeProvider{id: domain.ProviderGroq, models: []string{"whisper-large-v3"}},
		},
	})
	got, err := svc.ListModels(domain.ProviderGroq)
	require.NoError(t, err)
	require.Equal(t, []string{"whisper-large-v3"}, got)
}

func TestService_ListModels_UnknownProvider(t *testing.T) {
	svc := New(Deps{Providers: map[domain.ProviderID]ports.Provider{}})
	_, err := svc.ListModels(domain.ProviderGroq)
	require.ErrorIs(t, err, domain.ErrProviderMissing)
}
