// Package integration provides shared helpers for //go:build integration tests
// across adapter packages. Loads API keys via the same chain the runtime uses
// (user TOML → repo-local .transcribe.toml → env vars), so adding a key to
// .transcribe.toml at the repo root automatically enables matching tests.
package integration

import (
	"testing"

	"github.com/leotulipan/transcribe/internal/adapters/config"
	"github.com/leotulipan/transcribe/internal/core/domain"
)

// Key returns the API key for the given provider. Calls t.Skip if not configured.
func Key(t *testing.T, id domain.ProviderID) string {
	t.Helper()
	cfg, err := config.New().Load()
	if err != nil {
		t.Skipf("config load failed: %v", err)
	}
	k := cfg.APIKeys[id]
	if k == "" {
		t.Skipf("no API key configured for %s (set in .transcribe.toml or env)", id)
	}
	return k
}
