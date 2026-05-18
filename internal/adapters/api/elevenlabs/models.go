package elevenlabs

import (
	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

// Per https://elevenlabs.io/docs/api-reference/speech-to-text/convert
// Hardcoded list is the fallback when DiscoveredModels isn't populated.
// Refresh via `transcribe discover-models --provider elevenlabs`.
//
// CAVEAT: ElevenLabs' GET /v1/models endpoint returns TTS (text-to-speech)
// and STS (speech-to-speech) models — NOT speech-to-text. The actual STT
// models live under a separate scribe_* line and are added manually here.
// The TTS entries below (eleven_v3, eleven_english_sts_v2,
// eleven_multilingual_v2, ...) appear in discovery for completeness but
// will fail at transcription time. For transcription, pick a scribe_* model.
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

// ttsCaps marks discovered TTS models so they show up in the dropdown but
// without claimed STT capabilities. The capability check rejects them for
// SRT outputs; Transcribe() against them returns an upstream API error.
var ttsCaps = ports.ModelCapabilities{
	WordTimestamps:    false,
	SegmentTimestamps: false,
	LanguageHint:      false,
}

var modelCaps = map[string]ports.ModelCapabilities{
	// Actual STT models — use these for transcription.
	"scribe_v1":              sttCaps,
	"scribe_v1_experimental": sttCaps,

	// TTS / STS models surfaced via /v1/models. Listed so the dropdown
	// shows what discovery returns; selecting one for transcription will
	// fail with an upstream API error.
	"eleven_v3":                  ttsCaps,
	"eleven_english_sts_v2":      ttsCaps,
	"eleven_multilingual_sts_v2": ttsCaps,
	"eleven_multilingual_v1":     ttsCaps,
	"eleven_multilingual_v2":     ttsCaps,
	"eleven_monolingual_v1":      ttsCaps,
	"eleven_turbo_v2":            ttsCaps,
	"eleven_turbo_v2_5":          ttsCaps,
	"eleven_flash_v2":            ttsCaps,
	"eleven_flash_v2_5":          ttsCaps,
}

func Models() []string {
	out := make([]string, 0, len(modelCaps))
	for k := range modelCaps {
		out = append(out, k)
	}
	return out
}

func DefaultModel() string { return "scribe_v1" }

func Capabilities(model string) ports.ModelCapabilities {
	return modelCaps[model] // zero value if unknown — fail-safe
}
