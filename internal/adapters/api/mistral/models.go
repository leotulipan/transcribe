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

// Models returns the supported model IDs in best→worst order so UIs can
// surface the strongest option first. Map iteration would be random.
func Models() []string {
	return []string{
		"voxtral-small-latest",
		"voxtral-small-2507",
		"voxtral-mini-latest",
		"voxtral-mini-transcribe-2507",
		"voxtral-mini-2507",
	}
}

func DefaultModel() string { return "voxtral-small-latest" }

func Capabilities(model string) ports.ModelCapabilities {
	return modelCaps[model] // zero value if unknown — fail-safe
}
