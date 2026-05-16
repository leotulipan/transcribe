package config

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

func TestStore_RoundTrip(t *testing.T) {
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
	s := newWithPath(filepath.Join(t.TempDir(), "missing.toml"))
	out, err := s.Load()
	require.NoError(t, err)
	require.NotNil(t, out.APIKeys)
	require.Empty(t, out.APIKeys)
}

// Local alias so the test file doesn't import the ports package — we just need
// the same struct shape for round-trip checks.
type ports_Config = struct {
	APIKeys         map[domain.ProviderID]string
	DefaultProvider domain.ProviderID
	DefaultLanguage string
	FFmpegPath      string
}

// Test the OS-specific default path returns something non-empty.
func TestDefaultPath_NotEmpty(t *testing.T) {
	require.NotEmpty(t, defaultPath())
	_ = os.Getenv("LOCALAPPDATA") // touch to placate linters
}
