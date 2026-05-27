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
	Diarization:       true,
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

// Models returns the supported model IDs in best→worst order so UIs can
// surface the strongest option first. Map iteration would be random.
func Models() []string {
	return []string{"scribe_v2", "scribe_v1"}
}

func DefaultModel() string { return "scribe_v2" }

func Capabilities(model string) ports.ModelCapabilities {
	return modelCaps[model] // zero value if unknown — fail-safe
}
