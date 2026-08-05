package assemblyai

import (
	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

// Per https://www.assemblyai.com/docs/api-reference/transcripts
// AssemblyAI uses a two-step flow: upload then poll.
// Word timestamps are milliseconds in the response.
//
// Model selection: AssemblyAI's `speech_models` array accepts only
// `universal-3-5-pro` and `universal-2`. Anything else — including the
// retired `universal-3-pro`, the non-pro `universal-3`, `slam-1`, and the
// legacy singular `speech_model` values `best`/`nano` — is rejected with
// HTTP 400.
//
// Omitting `speech_models` entirely is equivalent to sending
// ["universal-3-5-pro", "universal-2"]: AssemblyAI runs its current flagship
// and falls back to universal-2 for languages outside the flagship's 18. That
// is what AutoModel expresses, and it is the default so we never have to ship
// a code change when AssemblyAI promotes a new flagship.
// See https://www.assemblyai.com/docs/pre-recorded-audio/select-the-speech-model

// AutoModel is a local sentinel, never an API value. When it is the selected
// model and no explicit fallback list was given, the client omits the
// speech_models key from the request body.
const AutoModel = "auto"

// deprecatedModels maps retired model IDs to their replacement. AssemblyAI
// retires universal-3-pro on 2026-09-02; requests naming it error after that.
var deprecatedModels = map[string]string{
	"universal-3-pro": "universal-3-5-pro",
}

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

// AutoModel advertises the same capabilities as the concrete models because
// the whole Universal family is uniform on these axes; if that ever stops
// being true, auto must advertise the intersection.
var modelCaps = map[string]ports.ModelCapabilities{
	AutoModel:           sttCaps,
	"universal-3-5-pro": sttCaps,
	"universal-2":       sttCaps,
}

// Models returns the supported model IDs with the default first, so UIs that
// pick models[0] land on the recommended selection. `auto` lets AssemblyAI
// choose; `universal-3-5-pro` covers 18 languages at the highest accuracy;
// `universal-2` covers 99 languages and is the automatic fallback. Map
// iteration would be random, hence the explicit slice.
func Models() []string {
	return []string{
		AutoModel,
		"universal-3-5-pro",
		"universal-2",
	}
}

func DefaultModel() string { return AutoModel }

// fallbackModel is appended to every explicitly pinned speech_models array so
// a job still succeeds when the pinned model does not cover the audio's
// language. The GUI/TUI have no fallback picker, so this guarantees a
// resilient result for every explicit selection.
const fallbackModel = "universal-2"

// NormalizeModels rewrites deprecated model IDs, preserving order and length.
// It returns the cleaned list plus the original deprecated IDs that were
// replaced (nil when there were none), so callers can warn the user. Unknown
// IDs pass through untouched: AssemblyAI's 400 response already names the
// valid set, and silently rewriting an unrecognized string would hide typos.
// Pure and idempotent — safe to call from several layers.
func NormalizeModels(in []string) (out, remapped []string) {
	if len(in) == 0 {
		return in, nil
	}
	out = make([]string, len(in))
	for i, m := range in {
		if replacement, ok := deprecatedModels[m]; ok {
			out[i] = replacement
			remapped = append(remapped, m)
			continue
		}
		out[i] = m
	}
	return out, remapped
}

func Capabilities(model string) ports.ModelCapabilities {
	return modelCaps[model] // zero value if unknown — fail-safe
}
