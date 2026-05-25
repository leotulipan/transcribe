package cli

import (
	"fmt"
	"strings"

	"github.com/spf13/cobra"

	"github.com/leotulipan/transcribe/internal/adapters/audio"
	"github.com/leotulipan/transcribe/internal/adapters/config"
	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

// validProviderIDs is the canonical set of known provider identifiers.
var validProviderIDs = map[domain.ProviderID]bool{
	domain.ProviderGroq:       true,
	domain.ProviderOpenAI:     true,
	domain.ProviderAssemblyAI: true,
	domain.ProviderElevenLabs: true,
	domain.ProviderGemini:     true,
	domain.ProviderMistral:    true,
}

// parseAPIKey splits "<provider>:<key>" into its parts. Everything after the
// first colon is treated as the key, so keys containing colons are handled
// correctly.
func parseAPIKey(raw string) (domain.ProviderID, string, error) {
	idx := strings.Index(raw, ":")
	if idx < 0 {
		return "", "", fmt.Errorf("--api-key: expected format <provider>:<key>, got %q", raw)
	}
	providerStr := raw[:idx]
	key := raw[idx+1:]
	if key == "" {
		return "", "", fmt.Errorf("--api-key: key portion is empty in %q", raw)
	}
	id := domain.ProviderID(providerStr)
	if !validProviderIDs[id] {
		return "", "", fmt.Errorf("--api-key: unknown provider %q (valid: groq, openai, assemblyai, elevenlabs, gemini, mistral)", providerStr)
	}
	return id, key, nil
}

func newSetupCmd(d Deps) *cobra.Command {
	var (
		groq, openai, assembly, eleven, gemini, mistral string
		defaultProvider, defaultLang, ffmpegPath        string
		apiKey                                          string
	)
	cmd := &cobra.Command{
		Use:   "setup",
		Short: "Write the on-disk config non-interactively",
		RunE: func(_ *cobra.Command, _ []string) error {
			cfg := d.Config // start from currently loaded state
			if cfg.APIKeys == nil {
				cfg.APIKeys = map[domain.ProviderID]string{}
			}
			apply := func(id domain.ProviderID, v string) {
				if v != "" {
					cfg.APIKeys[id] = v
				}
			}
			apply(domain.ProviderGroq, groq)
			apply(domain.ProviderOpenAI, openai)
			apply(domain.ProviderAssemblyAI, assembly)
			apply(domain.ProviderElevenLabs, eleven)
			apply(domain.ProviderGemini, gemini)
			apply(domain.ProviderMistral, mistral)
			// --api-key wins over per-provider flags for the same provider.
			if apiKey != "" {
				id, key, err := parseAPIKey(apiKey)
				if err != nil {
					return err
				}
				cfg.APIKeys[id] = key
			}
			if defaultProvider != "" {
				cfg.DefaultProvider = domain.ProviderID(defaultProvider)
			}
			if defaultLang != "" {
				cfg.DefaultLanguage = defaultLang
			}
			if ffmpegPath != "" {
				resolved, err := audio.ResolveBinary(ffmpegPath, "ffmpeg")
				if err != nil {
					return fmt.Errorf("--ffmpeg-path: %w", err)
				}
				cfg.FFmpegPath = resolved
				if resolved != ffmpegPath {
					fmt.Println("ffmpeg resolved to", resolved)
				}
			}
			store := config.New()
			if err := store.Save(cfg); err != nil {
				return err
			}
			fmt.Println("wrote", store.Path())
			_ = ports.Config{} // keep import if linter complains
			return nil
		},
	}
	cmd.Flags().StringVar(&groq, "groq-key", "", "Groq API key")
	cmd.Flags().StringVar(&openai, "openai-key", "", "OpenAI API key")
	cmd.Flags().StringVar(&assembly, "assemblyai-key", "", "AssemblyAI API key")
	cmd.Flags().StringVar(&eleven, "elevenlabs-key", "", "ElevenLabs API key")
	cmd.Flags().StringVar(&gemini, "gemini-key", "", "Gemini API key")
	cmd.Flags().StringVar(&mistral, "mistral-key", "", "Mistral API key")
	cmd.Flags().StringVar(&defaultProvider, "default-provider", "", "Default provider id")
	cmd.Flags().StringVar(&defaultLang, "default-language", "", "Default language (ISO-639-1)")
	cmd.Flags().StringVar(&ffmpegPath, "ffmpeg-path", "", "Path to ffmpeg executable (empty = PATH lookup)")
	cmd.Flags().StringVar(&apiKey, "api-key", "", "set a single API key: <provider>:<key> (e.g. groq:sk_xxx)")
	return cmd
}
