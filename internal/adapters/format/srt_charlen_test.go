package format

import "testing"

// SRT character-per-line wrapping (greedy split so no rendered line exceeds N
// chars) is not yet implemented in Go. The current writer caps blocks at 7
// words with no awareness of rendered width.
// See docs/plans/2-feature-parity-completion.md Phase 2b.
//
// Python source: audio_transcribe/transcribe_helpers/output_formatters.py — wrap_text_for_srt
// Python tests:  tests/unit/test_formatters.py — wrap-by-chars cases

func TestSRTCharLen_WrapsAtBoundary(t *testing.T) {
	t.Skip("pending: --chars-per-line wrapping not yet ported — see Phase 2b")
	// expected behavior:
	//   given a block with words totalling > N chars,
	//   the block emits multiple lines, each <= N chars (counting spaces).
}

func TestSRTCharLen_NeverBreaksMidWord(t *testing.T) {
	t.Skip("pending: --chars-per-line wrapping not yet ported — see Phase 2b")
	// expected behavior:
	//   even if a single word exceeds N chars, it stays whole on its own line.
}

func TestSRTCharLen_ZeroDisablesWrapping(t *testing.T) {
	t.Skip("pending: --chars-per-line wrapping not yet ported — see Phase 2b")
	// expected behavior:
	//   MaxCharsPerLine == 0 → identical output to today's writer (single
	//   line per block, joined by spaces).
}
