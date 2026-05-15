package mistral

import (
	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

// Per https://docs.mistral.ai/capabilities/audio/
// Voxtral returns segment-level timestamps but NOT word-level timestamps in v1.
// WordTimestamps: false — Voxtral + SRT is rejected by the capability check.
var modelCaps = map[string]ports.ModelCapabilities{
	"voxtral-mini-latest": {
		WordTimestamps:    false,
		SegmentTimestamps: true,
		LanguageHint:      true,
		AcceptedInputs: []domain.AudioFormat{
			{Codec: "mp3"}, {Codec: "wav"}, {Codec: "flac"}, {Codec: "ogg"},
			{Codec: "opus"}, {Codec: "m4a"}, {Codec: "aac"}, {Codec: "webm"},
		},
	},
	"voxtral-small-latest": {
		WordTimestamps:    false,
		SegmentTimestamps: true,
		LanguageHint:      true,
		AcceptedInputs: []domain.AudioFormat{
			{Codec: "mp3"}, {Codec: "wav"}, {Codec: "flac"}, {Codec: "ogg"},
			{Codec: "opus"}, {Codec: "m4a"}, {Codec: "aac"}, {Codec: "webm"},
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

func DefaultModel() string { return "voxtral-mini-latest" }

func Capabilities(model string) ports.ModelCapabilities {
	return modelCaps[model] // zero value if unknown — fail-safe
}
