package mistral

import (
	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

// Per https://docs.mistral.ai/capabilities/audio/
// Voxtral returns segment-level timestamps but NOT word-level timestamps in v1.
// WordTimestamps: false — Voxtral + SRT is rejected by the capability check.
// Hardcoded list is the fallback when DiscoveredModels isn't populated.
// Refresh via `transcribe discover-models --provider mistral`.
var voxtralCaps = ports.ModelCapabilities{
	WordTimestamps:    false,
	SegmentTimestamps: true,
	LanguageHint:      true,
	AcceptedInputs: []domain.AudioFormat{
		{Codec: "mp3"}, {Codec: "wav"}, {Codec: "flac"}, {Codec: "ogg"},
		{Codec: "opus"}, {Codec: "m4a"}, {Codec: "aac"}, {Codec: "webm"},
	},
}

var modelCaps = map[string]ports.ModelCapabilities{
	"voxtral-mini-latest":          voxtralCaps,
	"voxtral-mini-2507":            voxtralCaps,
	"voxtral-mini-transcribe-2507": voxtralCaps,
	"voxtral-small-latest":         voxtralCaps,
	"voxtral-small-2507":           voxtralCaps,
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
