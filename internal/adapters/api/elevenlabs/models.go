package elevenlabs

import (
	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

// Per https://elevenlabs.io/docs/api-reference/speech-to-text/convert
var modelCaps = map[string]ports.ModelCapabilities{
	"scribe_v1": {
		WordTimestamps:    true,
		SegmentTimestamps: false,
		LanguageHint:      true,
		AcceptedInputs: []domain.AudioFormat{
			{Codec: "mp3"}, {Codec: "mp4"}, {Codec: "m4a"}, {Codec: "wav"},
			{Codec: "flac"}, {Codec: "ogg"}, {Codec: "webm"}, {Codec: "opus"},
			{Codec: "mpga"},
		},
	},
	"scribe_v1_experimental": {
		WordTimestamps:    true,
		SegmentTimestamps: false,
		LanguageHint:      true,
		AcceptedInputs: []domain.AudioFormat{
			{Codec: "mp3"}, {Codec: "mp4"}, {Codec: "m4a"}, {Codec: "wav"},
			{Codec: "flac"}, {Codec: "ogg"}, {Codec: "webm"}, {Codec: "opus"},
			{Codec: "mpga"},
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

func DefaultModel() string { return "scribe_v1" }

func Capabilities(model string) ports.ModelCapabilities {
	return modelCaps[model] // zero value if unknown — fail-safe
}
