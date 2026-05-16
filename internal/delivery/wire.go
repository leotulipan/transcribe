package delivery

import (
	"net/http"

	"github.com/leotulipan/transcribe/internal/adapters/api/assemblyai"
	"github.com/leotulipan/transcribe/internal/adapters/api/elevenlabs"
	"github.com/leotulipan/transcribe/internal/adapters/api/gemini"
	"github.com/leotulipan/transcribe/internal/adapters/api/groq"
	"github.com/leotulipan/transcribe/internal/adapters/api/mistral"
	"github.com/leotulipan/transcribe/internal/adapters/api/openai"
	"github.com/leotulipan/transcribe/internal/adapters/audio"
	"github.com/leotulipan/transcribe/internal/adapters/cache"
	"github.com/leotulipan/transcribe/internal/adapters/format"
	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/core/services"
	"github.com/leotulipan/transcribe/internal/ports"
)

// BuildService wires the adapters into a TranscribeService.
func BuildService(cfg ports.Config, log ports.Logger) (ports.TranscribeService, error) {
	ffmpeg, err := audio.New(cfg.FFmpegPath, "", log)
	if err != nil {
		return nil, err
	}

	httpClient := &http.Client{}
	providers := map[domain.ProviderID]ports.Provider{}
	if k := cfg.APIKeys[domain.ProviderGroq];       k != "" { providers[domain.ProviderGroq]       = groq.New(k, httpClient) }
	if k := cfg.APIKeys[domain.ProviderOpenAI];     k != "" { providers[domain.ProviderOpenAI]     = openai.New(k, httpClient) }
	if k := cfg.APIKeys[domain.ProviderAssemblyAI]; k != "" { providers[domain.ProviderAssemblyAI] = assemblyai.New(k, httpClient) }
	if k := cfg.APIKeys[domain.ProviderElevenLabs]; k != "" { providers[domain.ProviderElevenLabs] = elevenlabs.New(k, httpClient) }
	if k := cfg.APIKeys[domain.ProviderGemini];     k != "" { providers[domain.ProviderGemini]     = gemini.New(k, httpClient) }
	if k := cfg.APIKeys[domain.ProviderMistral];    k != "" { providers[domain.ProviderMistral]    = mistral.New(k, httpClient) }

	writers := map[domain.OutputFormat]ports.FormatWriter{
		domain.FormatText:       format.NewText(),
		domain.FormatSRT:        format.NewSRT(),
		domain.FormatDavinciSRT: format.NewDaVinci(),
	}

	return services.New(services.Deps{
		Providers: providers,
		Audio:     ffmpeg,
		Cache:     cache.New(),
		Writers:   writers,
		Log:       log,
	}), nil
}
