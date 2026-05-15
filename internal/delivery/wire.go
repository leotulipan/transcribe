package delivery

import (
	"net/http"

	"github.com/leotulipan/transcribe/internal/adapters/api/groq"
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

	httpClient := &http.Client{} // groq.New sets its own timeout
	providers := map[domain.ProviderID]ports.Provider{}
	if key := cfg.APIKeys[domain.ProviderGroq]; key != "" {
		providers[domain.ProviderGroq] = groq.New(key, httpClient)
	}

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
