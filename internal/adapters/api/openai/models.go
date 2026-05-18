package openai

import (
	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

// Per https://platform.openai.com/docs/api-reference/audio/createTranscription
// Hardcoded list is the fallback when DiscoveredModels isn't populated.
// Refresh via `transcribe discover-models --provider openai`.
// All STT-capable models share the same audio capabilities on OpenAI.
var sttCaps = ports.ModelCapabilities{
	WordTimestamps:    true,
	SegmentTimestamps: true,
	LanguageHint:      true,
	AcceptedInputs: []domain.AudioFormat{
		{Codec: "mp3"}, {Codec: "mp4"}, {Codec: "mpeg"}, {Codec: "mpga"},
		{Codec: "m4a"}, {Codec: "wav"}, {Codec: "webm"}, {Codec: "flac"},
		{Codec: "ogg"}, {Codec: "opus"},
	},
}

var modelCaps = map[string]ports.ModelCapabilities{
	"whisper-1":                sttCaps,
	"gpt-4o-transcribe":        sttCaps,
	"gpt-4o-transcribe-diarize": sttCaps,
	"gpt-4o-mini-transcribe":   sttCaps,
}

func Models() []string {
	out := make([]string, 0, len(modelCaps))
	for k := range modelCaps {
		out = append(out, k)
	}
	return out
}

func DefaultModel() string { return "whisper-1" }

func Capabilities(model string) ports.ModelCapabilities {
	return modelCaps[model] // zero value if unknown — fail-safe
}
