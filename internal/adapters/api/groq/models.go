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

func Models() []string {
	out := make([]string, 0, len(modelCaps))
	for k := range modelCaps {
		out = append(out, k)
	}
	return out
}

func DefaultModel() string { return "whisper-large-v3" }

func Capabilities(model string) ports.ModelCapabilities {
	return modelCaps[model] // zero value if unknown — fail-safe
}
