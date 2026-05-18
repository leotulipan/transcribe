package elevenlabs

import "testing"

// Diarization (speaker labelling) is hardcoded to false in the ElevenLabs
// multipart payload at client.go:140. Once Phase 3 wires Request.SpeakerLabels
// through to the client and the parser populates Word speaker IDs, these
// tests flip to real assertions.
// See docs/plans/2-feature-parity-completion.md Phase 3.
//
// Python source: audio_transcribe/utils/api/elevenlabs.py
// Python tests:  tests/unit/test_parsers.py (speaker fields)
//                tests/integration/test_api_integration_elevenlabs.py

func TestDiarize_RequestPayloadIncludesDiarizeFlag(t *testing.T) {
	t.Skip("pending: diarize flag hardcoded to false — see Phase 3")
	// expected behavior:
	//   given Request{SpeakerLabels: true},
	//   the multipart body sent to /v1/speech-to-text contains
	//   diarize=true (not the current hardcoded "false").
}

func TestDiarize_ParsePopulatesSpeakerOnWords(t *testing.T) {
	t.Skip("pending: speaker field on domain.Word not yet added — see Phase 3")
	// expected behavior:
	//   given an ElevenLabs response with speaker_id on each word,
	//   parse() produces Result.Words where each Word has a non-empty Speaker
	//   field (matching the response's speaker_id).
}

func TestDiarize_OptedOutLeavesSpeakerEmpty(t *testing.T) {
	t.Skip("pending: diarize flag hardcoded to false — see Phase 3")
	// expected behavior:
	//   Request{SpeakerLabels: false} → multipart diarize=false,
	//   parser leaves Word.Speaker == "".
}
