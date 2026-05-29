package assemblyai

import (
	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

// Per https://www.assemblyai.com/docs/api-reference/transcripts
// AssemblyAI uses a two-step flow: upload then poll.
// Word timestamps are milliseconds in the response.
//
// Model selection: AssemblyAI accepts either a single `speech_model` or an
// ordered `speech_models` array with automatic fallback. The Universal /
// SLAM family is the current recommendation; "best" and "nano" are kept so
// existing configs and broad language coverage still work.
// See https://www.assemblyai.com/docs/pre-recorded-audio/select-the-speech-model

var sharedInputs = []domain.AudioFormat{
	{Codec: "mp3"}, {Codec: "wav"}, {Codec: "mp4"}, {Codec: "m4a"},
	{Codec: "aac"}, {Codec: "flac"}, {Codec: "ogg"}, {Codec: "webm"},
	{Codec: "mov"}, {Codec: "mpeg"},
}

var sttCaps = ports.ModelCapabilities{
	WordTimestamps:    true,
	SegmentTimestamps: false,
	Diarization:       true,
	LanguageHint:      true,
	AcceptedInputs:    sharedInputs,
}

var modelCaps = map[string]ports.ModelCapabilities{
	"universal-3-pro": sttCaps,
	"universal-3":     sttCaps,
	"universal-2":     sttCaps,
	"slam-1":          sttCaps,
	"best":            sttCaps,
	"nano":            sttCaps,
}

// Models returns the supported model IDs in best→worst order so UIs can
// surface the strongest option first. Universal-3 family covers EN/ES/PT/
// FR/DE/IT; Universal-2 and the legacy "best" / "nano" provide fallback
// language coverage. Map iteration would be random.
func Models() []string {
	return []string{
		"universal-3-pro",
		"universal-3",
		"universal-2",
		"slam-1",
		"best",
		"nano",
	}
}

func DefaultModel() string { return "universal-3-pro" }

func Capabilities(model string) ports.ModelCapabilities {
	return modelCaps[model] // zero value if unknown — fail-safe
}
