package assemblyai

import (
	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

// Per https://www.assemblyai.com/docs/api-reference/transcripts
// AssemblyAI uses a two-step flow: upload then poll.
// Word timestamps are milliseconds in the response.
//
// Model selection: AssemblyAI's `speech_models` array accepts only the current
// Universal family — `universal-3-pro` and `universal-2`. Anything else (the
// non-pro `universal-3`, the deprecated `slam-1`, and the legacy singular
// `speech_model` values `best`/`nano`) is rejected with HTTP 400: "speech_models
// must be a non-empty list ... of: universal-3-pro, universal-2". We therefore
// advertise only those two so the dropdown can never submit an invalid model.
// See https://www.assemblyai.com/docs/pre-recorded-audio/universal-3-pro

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
	"universal-2":     sttCaps,
}

// Models returns the supported model IDs in best→worst order so UIs can
// surface the strongest option first. `universal-3-pro` covers EN/ES/PT/FR/DE/
// IT; `universal-2` provides the remaining language coverage and is also the
// automatic fallback. These are the only values AssemblyAI's speech_models
// array accepts. Map iteration would be random.
func Models() []string {
	return []string{
		"universal-3-pro",
		"universal-2",
	}
}

func DefaultModel() string { return "universal-3-pro" }

// fallbackModel is appended to every request's speech_models array so a job
// still succeeds if the chosen model is unavailable. The GUI/TUI have no
// fallback picker, so this guarantees a resilient default for every provider.
const fallbackModel = "universal-2"

func Capabilities(model string) ports.ModelCapabilities {
	return modelCaps[model] // zero value if unknown — fail-safe
}
