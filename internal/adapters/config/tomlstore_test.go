package config

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

func TestStore_RoundTrip(t *testing.T) {
	t.Chdir(t.TempDir()) // isolate from any repo-local .transcribe.toml
	dir := t.TempDir()
	s := newWithPath(filepath.Join(dir, "config.toml"))

	in := ports_Config{
		APIKeys: map[domain.ProviderID]string{
			domain.ProviderGroq: "gsk_xyz",
		},
		DefaultProvider: domain.ProviderGroq,
		DefaultLanguage: "en",
		FFmpegPath:      `C:\tools\ffmpeg.exe`,
	}
	require.NoError(t, s.Save(in))

	out, err := s.Load()
	require.NoError(t, err)
	require.Equal(t, in.APIKeys[domain.ProviderGroq], out.APIKeys[domain.ProviderGroq])
	require.Equal(t, in.DefaultProvider, out.DefaultProvider)
	require.Equal(t, in.DefaultLanguage, out.DefaultLanguage)
	require.Equal(t, in.FFmpegPath, out.FFmpegPath)
}

func TestStore_EnvOverride(t *testing.T) {
	t.Chdir(t.TempDir())
	dir := t.TempDir()
	s := newWithPath(filepath.Join(dir, "config.toml"))
	require.NoError(t, s.Save(ports_Config{
		APIKeys: map[domain.ProviderID]string{domain.ProviderGroq: "from_file"},
	}))

	t.Setenv("GROQ_API_KEY", "from_env")
	t.Setenv("TRANSCRIBE_FFMPEG_PATH", `C:\override\ffmpeg.exe`)

	out, err := s.Load()
	require.NoError(t, err)
	require.Equal(t, "from_env", out.APIKeys[domain.ProviderGroq])
	require.Equal(t, `C:\override\ffmpeg.exe`, out.FFmpegPath)
}

func TestStore_LoadMissingFileReturnsEmpty(t *testing.T) {
	t.Chdir(t.TempDir())
	s := newWithPath(filepath.Join(t.TempDir(), "missing.toml"))
	out, err := s.Load()
	require.NoError(t, err)
	require.NotNil(t, out.APIKeys)
	require.Empty(t, out.APIKeys)
}

func TestStore_LocalConfigOverridesUserConfig(t *testing.T) {
	// User-level TOML in one tempdir, repo-local .transcribe.toml in another
	// that we cd into. Repo-local should win.
	userDir := t.TempDir()
	repoDir := t.TempDir()
	t.Chdir(repoDir)

	s := newWithPath(filepath.Join(userDir, "config.toml"))
	require.NoError(t, s.Save(ports_Config{
		APIKeys:         map[domain.ProviderID]string{domain.ProviderGroq: "user_groq"},
		DefaultLanguage: "en",
	}))

	require.NoError(t, os.WriteFile(
		filepath.Join(repoDir, LocalConfigName),
		[]byte(`[api_keys]`+"\n"+`groq = "local_groq"`+"\n"+`openai = "local_openai"`+"\n"),
		0o600,
	))

	out, err := s.Load()
	require.NoError(t, err)
	require.Equal(t, "local_groq", out.APIKeys[domain.ProviderGroq])
	require.Equal(t, "local_openai", out.APIKeys[domain.ProviderOpenAI])
	require.Equal(t, "en", out.DefaultLanguage) // preserved from user config
}

// Local alias so the test file doesn't import the ports package — we just need
// the same struct shape for round-trip checks.
type ports_Config = struct {
	APIKeys          map[domain.ProviderID]string
	DefaultProvider  domain.ProviderID
	DefaultLanguage  string
	FFmpegPath       string
	DiscoveredModels map[domain.ProviderID][]string
}

// Test the OS-specific default path returns something non-empty.
func TestDefaultPath_NotEmpty(t *testing.T) {
	require.NotEmpty(t, defaultPath())
	_ = os.Getenv("LOCALAPPDATA") // touch to placate linters
}
