// Package discover holds tiny shared helpers for adapter DiscoverModels
// implementations. Lives under internal/api/internal so only api/* packages
// can import it.
package discover

import "sort"

// SortUnique returns the input sorted, deduplicated, with empty strings removed.
// Does not modify the input.
func SortUnique(in []string) []string {
	seen := make(map[string]struct{}, len(in))
	for _, s := range in {
		if s == "" {
			continue
		}
		seen[s] = struct{}{}
	}
	out := make([]string, 0, len(seen))
	for s := range seen {
		out = append(out, s)
	}
	sort.Strings(out)
	return out
}
