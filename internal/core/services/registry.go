package services

import (
	"fmt"

	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

func providerFor(deps Deps, id domain.ProviderID) (ports.Provider, error) {
	p, ok := deps.Providers[id]
	if !ok {
		return nil, fmt.Errorf("%w: %s", domain.ErrProviderMissing, id)
	}
	return p, nil
}
