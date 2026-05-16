package config

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"

	"github.com/pelletier/go-toml/v2"

	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

// envKeys maps provider IDs to their canonical SDK environment variable names.
// These match the names each provider's own client libraries and CLI tools use,
// so a user's existing .env file (e.g., from the OpenAI, Anthropic, or Groq
// Python SDKs) works without modification.
var envKeys = map[domain.ProviderID]string{
	domain.ProviderAssemblyAI: "ASSEMBLYAI_API_KEY",
	domain.ProviderElevenLabs: "ELEVENLABS_API_KEY",
	domain.ProviderGroq:       "GROQ_API_KEY",
	domain.ProviderOpenAI:     "OPENAI_API_KEY",
	domain.ProviderGemini:     "GEMINI_API_KEY",
	domain.ProviderMistral:    "MISTRAL_API_KEY",
}

const envFFmpegPath = "TRANSCRIBE_FFMPEG_PATH"

// fileShape is the on-disk TOML schema.
type fileShape struct {
	DefaultProvider string            `toml:"default_provider"`
	DefaultLanguage string            `toml:"default_language"`
	FFmpegPath      string            `toml:"ffmpeg_path"`
	APIKeys         map[string]string `toml:"api_keys"`
}

type Store struct {
	path string
}

// New returns a Store using the OS-default path.
func New() *Store {
	return newWithPath(defaultPath())
}

func newWithPath(p string) *Store {
	return &Store{path: p}
}

func defaultPath() string {
	if runtime.GOOS == "windows" {
		base := os.Getenv("LOCALAPPDATA")
		if base == "" {
			base = filepath.Join(os.Getenv("USERPROFILE"), "AppData", "Local")
		}
		return filepath.Join(base, "transcribe", "config.toml")
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".transcribe", "config.toml")
}

func (s *Store) Path() string { return s.path }

func (s *Store) Load() (ports.Config, error) {
	cfg := ports.Config{APIKeys: map[domain.ProviderID]string{}}

	data, err := os.ReadFile(s.path)
	switch {
	case errors.Is(err, fs.ErrNotExist):
		// OK, empty config
	case err != nil:
		return cfg, err
	default:
		var fs_ fileShape
		if err := toml.Unmarshal(data, &fs_); err != nil {
			return cfg, err
		}
		cfg.DefaultProvider = domain.ProviderID(fs_.DefaultProvider)
		cfg.DefaultLanguage = fs_.DefaultLanguage
		cfg.FFmpegPath = fs_.FFmpegPath
		for k, v := range fs_.APIKeys {
			cfg.APIKeys[domain.ProviderID(k)] = v
		}
	}

	// Env overrides: each provider's canonical SDK env var wins over the file.
	for id, env := range envKeys {
		if v := os.Getenv(env); v != "" {
			cfg.APIKeys[id] = v
		}
	}
	if v := os.Getenv(envFFmpegPath); v != "" {
		cfg.FFmpegPath = v
	}
	return cfg, nil
}

func (s *Store) Save(cfg ports.Config) error {
	fs_ := fileShape{
		DefaultProvider: string(cfg.DefaultProvider),
		DefaultLanguage: cfg.DefaultLanguage,
		FFmpegPath:      cfg.FFmpegPath,
		APIKeys:         map[string]string{},
	}
	for k, v := range cfg.APIKeys {
		fs_.APIKeys[string(k)] = v
	}

	data, err := toml.Marshal(fs_)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(s.path), 0o700); err != nil {
		return err
	}
	return os.WriteFile(s.path, data, 0o600)
}
