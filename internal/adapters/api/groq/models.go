package groq

import (
	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

// Per https://console.groq.com/docs/speech-to-text
var modelCaps = map[string]ports.ModelCapabilities{
	"whisper-large-v3": {
		WordTimestamps:    true,
		SegmentTimestamps: true,
		LanguageHint:      true,
		AcceptedInputs: []domain.AudioFormat{
			{Codec: "mp3"}, {Codec: "mp4"}, {Codec: "aac"},
			{Codec: "flac"}, {Codec: "ogg"}, {Codec: "opus"},
			{Codec: "pcm_s16le"}, {Codec: "wav"},
		},
	},
	"whisper-large-v3-turbo": {
		WordTimestamps:    true,
		SegmentTimestamps: true,
		LanguageHint:      true,
		AcceptedInputs: []domain.AudioFormat{
			{Codec: "mp3"}, {Codec: "mp4"}, {Codec: "aac"},
			{Codec: "flac"}, {Codec: "ogg"}, {Codec: "opus"},
			{Codec: "pcm_s16le"}, {Codec: "wav"},
		},
	},
}

// Models returns the supported model IDs in best→worst order so UIs can
// surface the strongest option first. Map iteration would be random.
func Models() []string {
	return []string{"whisper-large-v3", "whisper-large-v3-turbo"}
}

func DefaultModel() string { return "whisper-large-v3" }

func Capabilities(model string) ports.ModelCapabilities {
	return modelCaps[model] // zero value if unknown — fail-safe
}
