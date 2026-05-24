package services

import (
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

// Edge cases for applyDavinci that complement davinci_test.go. Ports
// tests/unit/test_text_processing.py cases not already covered.

func TestDavinciApply_CustomFillerListOverridesDefault(t *testing.T) {
	res := &domain.Result{
		Words: []domain.Word{
			{Text: "okay", Start: 100 * time.Millisecond, End: 200 * time.Millisecond},
			{Text: "um", Start: 250 * time.Millisecond, End: 350 * time.Millisecond},
			{Text: "fine", Start: 400 * time.Millisecond, End: 500 * time.Millisecond},
		},
	}
	applyDavinci(res, &domain.DaVinciOptions{
		FillerWords:            []string{"okay"},
		SilentPortionThreshold: 5 * time.Second, // suppress pauses for this case
	})

	require.Equal(t, "OKAY", res.Words[0].Text, "custom filler 'okay' must be uppercased")
	require.Equal(t, "um", res.Words[1].Text, "default 'um' must NOT match when custom list is set")
	require.Equal(t, "fine", res.Words[2].Text)
}

func TestDavinciApply_MatchesAreCaseInsensitive(t *testing.T) {
	res := &domain.Result{
		Words: []domain.Word{
			{Text: "Um", Start: 0, End: 100 * time.Millisecond},
			{Text: "UH", Start: 200 * time.Millisecond, End: 300 * time.Millisecond},
			{Text: "ÄHM", Start: 400 * time.Millisecond, End: 500 * time.Millisecond},
		},
	}
	applyDavinci(res, &domain.DaVinciOptions{
		SilentPortionThreshold: 5 * time.Second,
	})
	for _, w := range res.Words {
		require.Equal(t, strings.ToUpper(w.Text), w.Text, "all default fillers must be uppercased regardless of input case")
	}
}

func TestDavinciApply_StripsTrailingPunctuationBeforeMatching(t *testing.T) {
	res := &domain.Result{
		Words: []domain.Word{
			{Text: "um,", Start: 0, End: 100 * time.Millisecond},
			{Text: "uh.", Start: 200 * time.Millisecond, End: 300 * time.Millisecond},
		},
	}
	applyDavinci(res, &domain.DaVinciOptions{
		SilentPortionThreshold: 5 * time.Second,
	})
	require.Equal(t, "UM,", res.Words[0].Text, "trailing comma stripped for match but kept in output")
	require.Equal(t, "UH.", res.Words[1].Text, "trailing period stripped for match but kept in output")
}

func TestDavinciApply_PauseExactlyAtThresholdInserts(t *testing.T) {
	res := &domain.Result{
		Words: []domain.Word{
			{Text: "first", Start: 0, End: 500 * time.Millisecond},
			// gap == threshold exactly
			{Text: "second", Start: 2000 * time.Millisecond, End: 2500 * time.Millisecond},
		},
	}
	applyDavinci(res, &domain.DaVinciOptions{
		SilentPortionThreshold: 1500 * time.Millisecond,
	})
	require.Len(t, res.Words, 3, "pause marker inserted at gap == threshold (inclusive)")
	require.Equal(t, "(...)", res.Words[1].Text)
}

func TestDavinciApply_GapJustBelowThresholdDoesNotInsert(t *testing.T) {
	res := &domain.Result{
		Words: []domain.Word{
			{Text: "first", Start: 0, End: 500 * time.Millisecond},
			// gap == 1499ms, threshold == 1500ms
			{Text: "second", Start: 1999 * time.Millisecond, End: 2500 * time.Millisecond},
		},
	}
	applyDavinci(res, &domain.DaVinciOptions{
		SilentPortionThreshold: 1500 * time.Millisecond,
	})
	require.Len(t, res.Words, 2, "no pause when gap < threshold")
}

func TestDavinciApply_MultipleConsecutivePauses(t *testing.T) {
	res := &domain.Result{
		Words: []domain.Word{
			{Text: "a", Start: 0, End: 200 * time.Millisecond},
			// 2.0s gap
			{Text: "b", Start: 2200 * time.Millisecond, End: 2400 * time.Millisecond},
			// 2.0s gap
			{Text: "c", Start: 4400 * time.Millisecond, End: 4600 * time.Millisecond},
		},
	}
	applyDavinci(res, &domain.DaVinciOptions{
		SilentPortionThreshold: 1500 * time.Millisecond,
	})
	var pauseCount int
	for _, w := range res.Words {
		if w.Text == "(...)" {
			pauseCount++
		}
	}
	require.Equal(t, 2, pauseCount, "two separate pause markers expected")
}

func TestDavinciApply_CleanTextEmitsUnchanged(t *testing.T) {
	res := &domain.Result{
		Words: []domain.Word{
			{Text: "Hello", Start: 0, End: 500 * time.Millisecond},
			{Text: "world", Start: 600 * time.Millisecond, End: 1100 * time.Millisecond},
		},
	}
	original := make([]domain.Word, len(res.Words))
	copy(original, res.Words)
	applyDavinci(res, &domain.DaVinciOptions{
		SilentPortionThreshold: 5 * time.Second,
	})
	require.Equal(t, original, res.Words, "no fillers and no pauses → no mutation")
}

func TestDavinciApply_RemoveFillersDropsMatches(t *testing.T) {
	res := &domain.Result{
		Words: []domain.Word{
			{Text: "So", Start: 0, End: 100 * time.Millisecond},
			{Text: "um", Start: 150 * time.Millisecond, End: 250 * time.Millisecond},
			{Text: "yeah", Start: 300 * time.Millisecond, End: 400 * time.Millisecond},
		},
	}
	applyDavinci(res, &domain.DaVinciOptions{
		SilentPortionThreshold: 5 * time.Second,
		RemoveFillers:          true,
	})
	require.Len(t, res.Words, 2, "filler word must be dropped entirely")
	require.Equal(t, "So", res.Words[0].Text)
	require.Equal(t, "yeah", res.Words[1].Text)
}

func TestDavinciApply_RemoveFillersDropsAllDefaults(t *testing.T) {
	res := &domain.Result{
		Words: []domain.Word{
			{Text: "um", Start: 0, End: 100 * time.Millisecond},
			{Text: "hello", Start: 150 * time.Millisecond, End: 250 * time.Millisecond},
			{Text: "uh", Start: 300 * time.Millisecond, End: 400 * time.Millisecond},
			{Text: "there", Start: 450 * time.Millisecond, End: 550 * time.Millisecond},
		},
	}
	applyDavinci(res, &domain.DaVinciOptions{
		SilentPortionThreshold: 5 * time.Second,
		RemoveFillers:          true,
	})
	require.Len(t, res.Words, 2)
	require.Equal(t, "hello", res.Words[0].Text)
	require.Equal(t, "there", res.Words[1].Text)
}

func TestDavinciApply_SuppressFillerLinesTruePreservesCase(t *testing.T) {
	res := &domain.Result{
		Words: []domain.Word{
			{Text: "um", Start: 0, End: 100 * time.Millisecond},
			{Text: "uh", Start: 150 * time.Millisecond, End: 250 * time.Millisecond},
			{Text: "hello", Start: 300 * time.Millisecond, End: 400 * time.Millisecond},
		},
	}
	applyDavinci(res, &domain.DaVinciOptions{
		SilentPortionThreshold: 5 * time.Second,
		SuppressFillerLines:    true,
	})
	require.Equal(t, "um", res.Words[0].Text, "SuppressFillerLines=true must leave filler in original case")
	require.Equal(t, "uh", res.Words[1].Text, "SuppressFillerLines=true must leave filler in original case")
	require.Equal(t, "hello", res.Words[2].Text)
}

func TestDavinciApply_SuppressFillerLinesFalseUppercases(t *testing.T) {
	res := &domain.Result{
		Words: []domain.Word{
			{Text: "um", Start: 0, End: 100 * time.Millisecond},
			{Text: "hello", Start: 150 * time.Millisecond, End: 250 * time.Millisecond},
		},
	}
	applyDavinci(res, &domain.DaVinciOptions{
		SilentPortionThreshold: 5 * time.Second,
		SuppressFillerLines:    false,
	})
	require.Equal(t, "UM", res.Words[0].Text, "SuppressFillerLines=false (default) must uppercase the filler")
	require.Equal(t, "hello", res.Words[1].Text)
}

func TestDavinciApply_RemoveFillersTakesPrecedenceOverSuppressFillerLines(t *testing.T) {
	res := &domain.Result{
		Words: []domain.Word{
			{Text: "um", Start: 0, End: 100 * time.Millisecond},
			{Text: "hello", Start: 150 * time.Millisecond, End: 250 * time.Millisecond},
		},
	}
	applyDavinci(res, &domain.DaVinciOptions{
		SilentPortionThreshold: 5 * time.Second,
		RemoveFillers:          true,
		SuppressFillerLines:    false,
	})
	require.Len(t, res.Words, 1, "RemoveFillers must drop the word even when SuppressFillerLines=false")
	require.Equal(t, "hello", res.Words[0].Text)
}

func TestDavinciApply_RemoveFillers_PauseAfterRemovedFillerStillInserts(t *testing.T) {
	// real word → filler (removed) → long gap → real word
	// prevEnd must advance to the filler's End so the gap is measured from there.
	res := &domain.Result{
		Words: []domain.Word{
			{Text: "hello", Start: 0, End: 500 * time.Millisecond},
			{Text: "um", Start: 600 * time.Millisecond, End: 700 * time.Millisecond},
			{Text: "world", Start: 2300 * time.Millisecond, End: 2800 * time.Millisecond},
		},
	}
	applyDavinci(res, &domain.DaVinciOptions{
		SilentPortionThreshold: 1500 * time.Millisecond,
		RemoveFillers:          true,
	})
	// Expected: hello, (...), world — filler gone, pause marker present
	require.Len(t, res.Words, 3, "real word + pause marker + real word expected")
	require.Equal(t, "hello", res.Words[0].Text)
	require.Equal(t, "(...)", res.Words[1].Text, "pause marker must appear after the removed filler's end")
	require.Equal(t, "world", res.Words[2].Text)
}
