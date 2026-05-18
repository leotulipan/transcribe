package config

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"strings"

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

// LocalConfigName is the filename Load() looks for in (or above) the CWD as an
// override layer for the user-level config. Same shape as the main TOML;
// intended to carry per-checkout API keys for tests and dev runs.
// Added to .gitignore.
const LocalConfigName = ".transcribe.toml"

// fileShape is the on-disk TOML schema.
type fileShape struct {
	DefaultProvider  string              `toml:"default_provider"`
	DefaultLanguage  string              `toml:"default_language"`
	FFmpegPath       string              `toml:"ffmpeg_path"`
	APIKeys          map[string]string   `toml:"api_keys"`
	DiscoveredModels map[string][]string `toml:"discovered_models,omitempty"`
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

	// Layer 1: user-level TOML at s.path (e.g. %LOCALAPPDATA%\transcribe\config.toml).
	if err := mergeFile(s.path, &cfg); err != nil {
		return cfg, err
	}

	// Layer 2: repo-local override at .transcribe.toml in (or above) the CWD.
	// Walks up from CWD to find the file, so running from a subdirectory still
	// picks up the project's local keys.
	if local := findLocalConfig(); local != "" {
		if err := mergeFile(local, &cfg); err != nil {
			return cfg, err
		}
	}

	// Layer 3: env vars (each provider's canonical SDK var) win over both files.
	// TrimSpace handles CRLF/trailing-whitespace artifacts from .env loaders
	// that don't normalize line endings on Windows.
	for id, env := range envKeys {
		if v := strings.TrimSpace(os.Getenv(env)); v != "" {
			cfg.APIKeys[id] = v
		}
	}
	if v := strings.TrimSpace(os.Getenv(envFFmpegPath)); v != "" {
		cfg.FFmpegPath = v
	}
	return cfg, nil
}

// mergeFile reads a TOML config file and merges its values into cfg.
// Missing file is not an error.
func mergeFile(path string, cfg *ports.Config) error {
	data, err := os.ReadFile(path)
	switch {
	case errors.Is(err, fs.ErrNotExist):
		return nil
	case err != nil:
		return err
	}
	var fs_ fileShape
	if err := toml.Unmarshal(data, &fs_); err != nil {
		return err
	}
	if v := strings.TrimSpace(fs_.DefaultProvider); v != "" {
		cfg.DefaultProvider = domain.ProviderID(v)
	}
	if v := strings.TrimSpace(fs_.DefaultLanguage); v != "" {
		cfg.DefaultLanguage = v
	}
	if v := strings.TrimSpace(fs_.FFmpegPath); v != "" {
		cfg.FFmpegPath = v
	}
	for k, v := range fs_.APIKeys {
		if v := strings.TrimSpace(v); v != "" {
			cfg.APIKeys[domain.ProviderID(k)] = v
		}
	}
	if len(fs_.DiscoveredModels) > 0 {
		if cfg.DiscoveredModels == nil {
			cfg.DiscoveredModels = map[domain.ProviderID][]string{}
		}
		for k, list := range fs_.DiscoveredModels {
			if len(list) > 0 {
				cfg.DiscoveredModels[domain.ProviderID(k)] = list
			}
		}
	}
	return nil
}

// findLocalConfig walks up from the CWD looking for LocalConfigName. Returns
// the absolute path or "" if not found / on any error.
func findLocalConfig() string {
	dir, err := os.Getwd()
	if err != nil {
		return ""
	}
	for {
		candidate := filepath.Join(dir, LocalConfigName)
		if _, err := os.Stat(candidate); err == nil {
			return candidate
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return ""
		}
		dir = parent
	}
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
	if len(cfg.DiscoveredModels) > 0 {
		fs_.DiscoveredModels = map[string][]string{}
		for k, list := range cfg.DiscoveredModels {
			if len(list) > 0 {
				fs_.DiscoveredModels[string(k)] = list
			}
		}
	}

	data, err := toml.Marshal(fs_)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(s.path), 0o700); err != nil {
		return err
	}
	// Atomic write: write to a temp file in the same dir, then rename.
	// Both the GUI settings save and `transcribe discover-models` may write
	// the same file; this avoids torn writes if they race.
	tmp, err := os.CreateTemp(filepath.Dir(s.path), ".config.toml.tmp.*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName) // best-effort cleanup on error paths
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Chmod(0o600); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpName, s.path)
}
