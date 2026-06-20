package cli

import (
	"bytes"
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

// ---------------------------------------------------------------------------
// --list flag on transcribe subcommand
// ---------------------------------------------------------------------------

type stubService struct {
	providers []domain.ProviderID
	models    map[domain.ProviderID][]string
}

func (s *stubService) ListProviders() []domain.ProviderID { return s.providers }
func (s *stubService) DefaultModel(id domain.ProviderID) string {
	if ms := s.models[id]; len(ms) > 0 {
		return ms[0]
	}
	return ""
}
func (s *stubService) ListModels(id domain.ProviderID) ([]string, error) {
	return s.models[id], nil
}
func (s *stubService) DiscoverModels(_ context.Context, _ domain.ProviderID) ([]string, error) {
	return nil, nil
}
func (s *stubService) Submit(_ context.Context, _ domain.Request) (ports.Job, error) {
	return nil, nil
}
func (s *stubService) Capabilities(_ domain.ProviderID, _ string) (ports.ModelCapabilities, bool) {
	return ports.ModelCapabilities{}, true
}

func newStubDeps() Deps {
	svc := &stubService{
		providers: []domain.ProviderID{domain.ProviderGroq, domain.ProviderOpenAI},
		models: map[domain.ProviderID][]string{
			domain.ProviderGroq:   {"whisper-large-v3"},
			domain.ProviderOpenAI: {"whisper-1"},
		},
	}
	return Deps{Service: svc}
}

func TestTranscribeCmd_ListPrintsProvidersAndExits(t *testing.T) {
	d := newStubDeps()
	cmd := newTranscribeCmd(d)
	buf := &bytes.Buffer{}
	cmd.SetOut(buf)
	// Redirect the os.Stdout writes: printProviders writes to os.Stdout by default
	// when called from transcribe's RunE. We capture via the command's stdout only
	// when the command is invoked through the root (which sets out). For a direct
	// unit test we invoke RunE directly and check that it returns nil.
	err := cmd.RunE(cmd, []string{})
	// When --list is not set, RunE returns EscalateToTUI or a similar error because
	// no files are given. Set the flag first.
	require.NoError(t, cmd.Flags().Set("list", "true"))
	err = cmd.RunE(cmd, []string{})
	require.NoError(t, err, "--list must return nil even with no input files")
}

func TestTranscribeCmd_HasListFlag(t *testing.T) {
	cmd := newTranscribeCmd(Deps{})
	f := cmd.Flags().Lookup("list")
	require.NotNil(t, f, "--list flag must be registered")
	require.Equal(t, "false", f.DefValue)
}

// ---------------------------------------------------------------------------
// --api-key flag on setup subcommand
// ---------------------------------------------------------------------------

func TestSetupCmd_HasApiKeyFlag(t *testing.T) {
	cmd := newSetupCmd(Deps{})
	f := cmd.Flags().Lookup("api-key")
	require.NotNil(t, f, "--api-key flag must be registered")
	require.Equal(t, "", f.DefValue)
}

func TestSetupCmd_ApiKeyValidatesProvider(t *testing.T) {
	_, _, err := parseAPIKey("bogus:somekey")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unknown provider")
}

func TestSetupCmd_ApiKeyWithColonInValue(t *testing.T) {
	id, key, err := parseAPIKey("groq:sk:abc")
	require.NoError(t, err)
	require.Equal(t, domain.ProviderGroq, id)
	require.Equal(t, "sk:abc", key, "everything after the first colon must be the key")
}

func TestSetupCmd_ApiKeyParsesValidProvider(t *testing.T) {
	providers := []string{"groq", "openai", "assemblyai", "elevenlabs", "gemini", "mistral"}
	for _, p := range providers {
		t.Run(p, func(t *testing.T) {
			id, key, err := parseAPIKey(p + ":sk_xxx")
			require.NoError(t, err)
			require.Equal(t, domain.ProviderID(p), id)
			require.Equal(t, "sk_xxx", key)
		})
	}
}

func TestSetupCmd_ApiKeyRejectsMissingColon(t *testing.T) {
	_, _, err := parseAPIKey("groqsk_xxx")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "expected format")
}

func TestSetupCmd_ApiKeyRejectsEmptyKey(t *testing.T) {
	_, _, err := parseAPIKey("groq:")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "empty")
}

// ---------------------------------------------------------------------------
// parseAPIKey unit coverage
// ---------------------------------------------------------------------------

func TestParseAPIKey_AllKnownProviders(t *testing.T) {
	cases := []struct {
		raw      string
		wantID   domain.ProviderID
		wantKey  string
	}{
		{"groq:sk_xxx", domain.ProviderGroq, "sk_xxx"},
		{"openai:sk_xxx", domain.ProviderOpenAI, "sk_xxx"},
		{"assemblyai:sk_xxx", domain.ProviderAssemblyAI, "sk_xxx"},
		{"elevenlabs:sk_xxx", domain.ProviderElevenLabs, "sk_xxx"},
		{"gemini:sk_xxx", domain.ProviderGemini, "sk_xxx"},
		{"mistral:sk_xxx", domain.ProviderMistral, "sk_xxx"},
	}
	for _, tc := range cases {
		t.Run(tc.raw, func(t *testing.T) {
			id, key, err := parseAPIKey(tc.raw)
			require.NoError(t, err)
			require.Equal(t, tc.wantID, id)
			require.Equal(t, tc.wantKey, key)
		})
	}
}
