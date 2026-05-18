package ports

import "context"

// ModelDiscoverer is an OPTIONAL capability for Provider implementations
// that expose a live "list models" endpoint. Adapters whose upstream API
// has no such endpoint simply don't implement it; callers must use a type
// assertion (`p.(ports.ModelDiscoverer)`) and skip on miss.
//
// Implementations should use a non-consuming HTTP request, validate the API
// key (a 401/403 from upstream is a real error to surface), and return a
// sorted unique slice of model IDs.
type ModelDiscoverer interface {
	DiscoverModels(ctx context.Context) ([]string, error)
}
