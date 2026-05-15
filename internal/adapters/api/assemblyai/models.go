package assemblyai

import (
	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

// Per https://www.assemblyai.com/docs/api-reference/transcripts
// AssemblyAI uses a two-step flow: upload then poll.
// Word timestamps are milliseconds in the response.
var modelCaps = map[string]ports.ModelCapabilities{
	"best": {
		WordTimestamps:    true,
		SegmentTimestamps: false,
		LanguageHint:      true,
		AcceptedInputs: []domain.AudioFormat{
			{Codec: "mp3"}, {Codec: "wav"}, {Codec: "mp4"}, {Codec: "m4a"},
			{Codec: "aac"}, {Codec: "flac"}, {Codec: "ogg"}, {Codec: "webm"},
			{Codec: "mov"}, {Codec: "mpeg"},
		},
	},
	"nano": {
		WordTimestamps:    true,
		SegmentTimestamps: false,
		LanguageHint:      true,
		AcceptedInputs: []domain.AudioFormat{
			{Codec: "mp3"}, {Codec: "wav"}, {Codec: "mp4"}, {Codec: "m4a"},
			{Codec: "aac"}, {Codec: "flac"}, {Codec: "ogg"}, {Codec: "webm"},
			{Codec: "mov"}, {Codec: "mpeg"},
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

func DefaultModel() string { return "best" }

func Capabilities(model string) ports.ModelCapabilities {
	return modelCaps[model] // zero value if unknown — fail-safe
}
