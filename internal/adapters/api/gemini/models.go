package gemini

import (
	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

// Per https://ai.google.dev/gemini-api/docs/audio
// Gemini does NOT natively return word-level timestamps — only unstructured text.
// WordTimestamps: false for all current Gemini models.
// This means Gemini + SRT / DaVinciSRT is rejected by the service's capability check.
// LanguageHint: false — Gemini auto-detects language from the prompt/audio context.
// Hardcoded list is the fallback when DiscoveredModels isn't populated.
// Refresh via `transcribe discover-models --provider gemini`.
var geminiCaps = ports.ModelCapabilities{
	WordTimestamps:    false,
	SegmentTimestamps: false,
	LanguageHint:      false,
	AcceptedInputs: []domain.AudioFormat{
		{Codec: "wav"}, {Codec: "mp3"}, {Codec: "aiff"}, {Codec: "aac"},
		{Codec: "ogg"}, {Codec: "flac"},
	},
}

var modelCaps = map[string]ports.ModelCapabilities{
	"gemini-2.0-flash":          geminiCaps,
	"gemini-2.5-flash":          geminiCaps,
	"gemini-2.5-flash-lite":     geminiCaps,
	"gemini-2.5-pro":            geminiCaps,
	"gemini-3-flash-preview":    geminiCaps,
	"gemini-3-pro-preview":      geminiCaps,
	"gemini-3.1-flash-lite":     geminiCaps,
	"gemini-3.1-pro-preview":    geminiCaps,
	"gemini-flash-latest":       geminiCaps,
	"gemini-pro-latest":         geminiCaps,
}

func Models() []string {
	out := make([]string, 0, len(modelCaps))
	for k := range modelCaps {
		out = append(out, k)
	}
	return out
}

func DefaultModel() string { return "gemini-2.5-flash" }

func Capabilities(model string) ports.ModelCapabilities {
	return modelCaps[model] // zero value if unknown — fail-safe
}
