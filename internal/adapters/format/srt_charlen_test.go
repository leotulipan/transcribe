package format

import (
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

// SRT character-per-line wrapping (greedy split so no rendered line exceeds N
// chars) is not yet implemented in Go. The current writer caps blocks at 7
// words with no awareness of rendered width.
// See docs/plans/2-feature-parity-completion.md Phase 2b.
//
// Python source: audio_transcribe/transcribe_helpers/output_formatters.py — wrap_text_for_srt
// Python tests:  tests/unit/test_formatters.py — wrap-by-chars cases

func TestSRTCharLen_WrapsAtBoundary(t *testing.T) {
	// given a block with words totalling > N chars,
	// the block emits multiple lines, each <= N chars (counting spaces).
	const maxChars = 10
	words := []domain.Word{
		{Text: "Hello", Start: 0, End: 500 * time.Millisecond},
		{Text: "world", Start: 600 * time.Millisecond, End: 1100 * time.Millisecond},
		{Text: "foo", Start: 1200 * time.Millisecond, End: 1500 * time.Millisecond},
	}
	// "Hello world foo" = 15 chars, maxChars=10 → must wrap.
	// Greedy: "Hello" (5) + " world" = 11 > 10 → line 1 = "Hello"
	//          "world" (5) + " foo" = 9 ≤ 10 → line 2 = "world foo"
	lines := wrapByChars(words, maxChars)
	require.Len(t, lines, 2, "should produce 2 lines when greedy fill forces a break")
	for _, line := range lines {
		rendered := joinTexts(line)
		require.LessOrEqual(t, len(rendered), maxChars,
			"each line must be <= maxChars; got %q", rendered)
	}
}

func TestSRTCharLen_NeverBreaksMidWord(t *testing.T) {
	// even if a single word exceeds N chars, it stays whole on its own line.
	const maxChars = 4
	words := []domain.Word{
		{Text: "Extraordinaire", Start: 0, End: 1 * time.Second},
	}
	lines := wrapByChars(words, maxChars)
	require.Len(t, lines, 1, "single oversized word must stay whole on its own line")
	require.Len(t, lines[0], 1)
	require.Equal(t, "Extraordinaire", lines[0][0].Text)
}

func TestSRTCharLen_ZeroDisablesWrapping(t *testing.T) {
	// MaxCharsPerLine == 0 → identical output to today's writer (single
	// line per block, joined by spaces).
	words := mkWords([]string{"Hello", "world", "this", "is", "a", "test"})

	// Write with opts zero (no wrapping)
	res := &domain.Result{Words: words}
	var lines []string
	blocks := groupWords(res.Words, srtMaxWordsPerBlock, srtMaxGap)
	for _, blk := range blocks {
		wrapped := wrapByChars(blk.Words, 0)
		require.Len(t, wrapped, 1, "maxChars=0 must produce exactly one line per block")
		lines = append(lines, joinTexts(wrapped[0]))
	}

	// Verify against what today's writer would produce (naive join).
	for bi, blk := range blocks {
		want := joinTexts(blk.Words)
		require.Equal(t, want, lines[bi], "zero maxChars must match naive join")
	}
}

func joinTexts(words []domain.Word) string {
	parts := make([]string, len(words))
	for i, w := range words {
		parts[i] = w.Text
	}
	return strings.Join(parts, " ")
}
