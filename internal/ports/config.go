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
}
