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
var modelCaps = map[string]ports.ModelCapabilities{
	"gemini-2.0-flash": {
		WordTimestamps:    false,
		SegmentTimestamps: false,
		LanguageHint:      false,
		AcceptedInputs: []domain.AudioFormat{
			{Codec: "wav"}, {Codec: "mp3"}, {Codec: "aiff"}, {Codec: "aac"},
			{Codec: "ogg"}, {Codec: "flac"},
		},
	},
	"gemini-2.5-flash": {
		WordTimestamps:    false,
		SegmentTimestamps: false,
		LanguageHint:      false,
		AcceptedInputs: []domain.AudioFormat{
			{Codec: "wav"}, {Codec: "mp3"}, {Codec: "aiff"}, {Codec: "aac"},
			{Codec: "ogg"}, {Codec: "flac"},
		},
	},
	"gemini-2.5-pro": {
		WordTimestamps:    false,
		SegmentTimestamps: false,
		LanguageHint:      false,
		AcceptedInputs: []domain.AudioFormat{
			{Codec: "wav"}, {Codec: "mp3"}, {Codec: "aiff"}, {Codec: "aac"},
			{Codec: "ogg"}, {Codec: "flac"},
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

func DefaultModel() string { return "gemini-2.0-flash" }

func Capabilities(model string) ports.ModelCapabilities {
	return modelCaps[model] // zero value if unknown — fail-safe
}
