package ports

import "github.com/leotulipan/transcribe/internal/core/domain"

type ResultCache interface {
    Lookup(inputPath string, p domain.ProviderID) (*domain.Result, bool, error)
    Save(inputPath string, r *domain.Result) error
}
