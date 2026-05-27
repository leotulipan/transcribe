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
		Diarization:       true,
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
		Diarization:       true,
		LanguageHint:      true,
		AcceptedInputs: []domain.AudioFormat{
			{Codec: "mp3"}, {Codec: "wav"}, {Codec: "mp4"}, {Codec: "m4a"},
			{Codec: "aac"}, {Codec: "flac"}, {Codec: "ogg"}, {Codec: "webm"},
			{Codec: "mov"}, {Codec: "mpeg"},
		},
	},
}

// Models returns the supported model IDs in best→worst order so UIs can
// surface the strongest option first. Map iteration would be random.
func Models() []string {
	return []string{"best", "nano"}
}

func DefaultModel() string { return "best" }

func Capabilities(model string) ports.ModelCapabilities {
	return modelCaps[model] // zero value if unknown — fail-safe
}
