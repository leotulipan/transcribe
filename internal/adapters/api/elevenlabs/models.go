package elevenlabs

import (
	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

// ElevenLabs Speech-to-Text models.
// Per https://elevenlabs.io/docs/api-reference/speech-to-text/convert
//
// Three STT model IDs are currently supported:
//   - scribe_v1            legacy batch transcription model
//   - scribe_v2            current batch transcription model (default)
//   - scribe_v2_realtime   live WebSocket streaming model — NOT yet wired up
//                          in this binary (we only do batch POST today). It's
//                          omitted from modelCaps so the dropdown only shows
//                          models the user can actually run from here. Add it
//                          back when streaming support lands.
//
// Note on discovery: ElevenLabs' GET /v1/models endpoint returns TTS and STS
// (speech-to-speech) voice models — NOT the scribe_* STT models. So a
// `transcribe discover-models --provider elevenlabs` run will populate the
// config with eleven_* TTS IDs that aren't usable for transcription. Until
// upstream surfaces scribe_* via a dedicated endpoint, this hardcoded
// fallback is the source of truth for what works.
var sttCaps = ports.ModelCapabilities{
	WordTimestamps:    true,
	SegmentTimestamps: false,
	LanguageHint:      true,
	AcceptedInputs: []domain.AudioFormat{
		{Codec: "mp3"}, {Codec: "mp4"}, {Codec: "m4a"}, {Codec: "wav"},
		{Codec: "flac"}, {Codec: "ogg"}, {Codec: "webm"}, {Codec: "opus"},
		{Codec: "mpga"},
	},
}

var modelCaps = map[string]ports.ModelCapabilities{
	"scribe_v1": sttCaps,
	"scribe_v2": sttCaps,
}

func Models() []string {
	out := make([]string, 0, len(modelCaps))
	for k := range modelCaps {
		out = append(out, k)
	}
	return out
}

func DefaultModel() string { return "scribe_v2" }

func Capabilities(model string) ports.ModelCapabilities {
	return modelCaps[model] // zero value if unknown — fail-safe
}
