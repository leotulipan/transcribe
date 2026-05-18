package services

import "testing"

// Output-format fallback (e.g. caller requests "srt" but provider returned no
// word timing → fall back to text and warn) is not yet implemented in Go.
// These tests lock the contract before the feature lands.
//
// Python source: implicit in audio_transcribe/cli.py + utils/formatters.py
// Python tests:  tests/integration/test_output_format_fallback.py

func TestFormatFallback_SRTFallsBackToTextWithoutTimestamps(t *testing.T) {
	t.Skip("pending: output-format fallback logic not yet ported")
	// expected behavior:
	//   given Result with empty Words but non-empty Text,
	//   requested formats [FormatSRT],
	//   pipeline writes a .txt file and emits a warning event.
}

func TestFormatFallback_DavinciFallsBackToSRTWhenNoFillerWords(t *testing.T) {
	t.Skip("pending: output-format fallback logic not yet ported")
	// expected behavior:
	//   given Result words with no filler words and no pauses long enough,
	//   davinci_srt output is still produced (no fallback needed); fallback
	//   only triggers when Words is empty entirely.
}

func TestFormatFallback_RespectsExplicitFormatList(t *testing.T) {
	t.Skip("pending: output-format fallback logic not yet ported")
	// expected behavior:
	//   if user asks for text + srt and srt can't be produced,
	//   text still emits and srt is skipped (not duplicated as text).
}
