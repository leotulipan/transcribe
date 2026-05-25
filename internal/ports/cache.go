package ports

import "github.com/leotulipan/transcribe/internal/core/domain"

type ResultCache interface {
	Lookup(inputPath string, p domain.ProviderID) (*domain.Result, bool, error)
	Save(inputPath string, r *domain.Result) error
	// LoadFromFile reads an arbitrary sidecar JSON path and returns the parsed
	// Result. Unlike Lookup it does not construct the path — the caller provides
	// the exact file. Returns an error when the file is missing or malformed.
	LoadFromFile(jsonPath string) (*domain.Result, error)
}
