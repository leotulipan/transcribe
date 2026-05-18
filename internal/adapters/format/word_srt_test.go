package format

import "testing"

// Word-level SRT output (one subtitle per word, useful for tight caption sync)
// is not yet implemented in Go. domain.FormatWordSRT does not exist yet.
// See docs/plans/2-feature-parity-completion.md Phase 2a.
//
// Python source: audio_transcribe/utils/formatters.py — create_srt(format_type="word")
// Python tests:  tests/unit/test_formatters.py — word-SRT cases

func TestWordSRT_OneSubtitlePerWord(t *testing.T) {
	t.Skip("pending: FormatWordSRT not yet defined — see Phase 2a")
	// expected behavior:
	//   given Words [{"Hello", 1.0–1.5s}, {"world", 1.6–2.1s}],
	//   output is:
	//     1
	//     00:00:01,000 --> 00:00:01,500
	//     Hello
	//
	//     2
	//     00:00:01,600 --> 00:00:02,100
	//     world
}

func TestWordSRT_EmptyResultProducesEmptyFile(t *testing.T) {
	t.Skip("pending: FormatWordSRT not yet defined — see Phase 2a")
	// expected behavior:
	//   Result with no Words → output file is empty (no header, no error).
}

func TestWordSRT_PreservesPunctuationOnWord(t *testing.T) {
	t.Skip("pending: FormatWordSRT not yet defined — see Phase 2a")
	// expected behavior:
	//   Word.Text "Hello," renders as "Hello," (comma not stripped).
}
