package format

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

func TestGroupWords_RespectsMaxWords(t *testing.T) {
	words := mkWords([]string{"one", "two", "three", "four", "five", "six", "seven", "eight"})
	blocks := groupWords(words, 7, 10*time.Second)
	require.GreaterOrEqual(t, len(blocks), 2)
	require.LessOrEqual(t, len(blocks[0].Words), 7)
}

func TestGroupWords_BreaksOnLongGap(t *testing.T) {
	words := []domain.Word{
		{Text: "hello", Start: 0, End: 500 * time.Millisecond},
		// big silent gap
		{Text: "world", Start: 5 * time.Second, End: 5500 * time.Millisecond},
	}
	blocks := groupWords(words, 7, 3*time.Second)
	require.Len(t, blocks, 2, "long gap should force a block break")
}

func mkWords(texts []string) []domain.Word {
	out := make([]domain.Word, len(texts))
	for i, t := range texts {
		start := time.Duration(i) * 400 * time.Millisecond
		out[i] = domain.Word{Text: t, Start: start, End: start + 300*time.Millisecond}
	}
	return out
}
