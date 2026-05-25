package ports

import "github.com/leotulipan/transcribe/internal/core/domain"

type ConfigStore interface {
    Load() (Config, error)
    Save(Config) error
    Path() string
}

type Config struct {
    APIKeys         map[domain.ProviderID]string
    DefaultProvider domain.ProviderID
    DefaultLanguage string
    FFmpegPath      string // empty = exec.LookPath("ffmpeg")
    // DiscoveredModels caches per-provider model lists obtained from each
    // provider's live "list models" endpoint via `transcribe discover-models`
    // or the GUI refresh action. Empty = use the adapter's hardcoded fallback.
    DiscoveredModels map[domain.ProviderID][]string
}

// SetAPIKey stores or replaces the API key for the given provider.
func (c *Config) SetAPIKey(p domain.ProviderID, key string) {
    if c.APIKeys == nil {
        c.APIKeys = map[domain.ProviderID]string{}
    }
    c.APIKeys[p] = key
}
