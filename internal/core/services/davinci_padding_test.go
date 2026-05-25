package services

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

// --padding-start subtracts a small amount from each word's start time so the
// DaVinci-imported subtitle leads its audio (compensates editor preview lag).
//
// Python source: audio_transcribe/transcribe_helpers/output_formatters.py — apply_intelligent_padding
// Python tests:  tests/unit/test_text_processing.py — padding cases

func TestDavinciPadding_SubtractsFromStart(t *testing.T) {
	// Words separated by 200ms gaps; padding is 50ms → gap/2 = 100ms > 50ms,
	// so the full padding is subtracted from every word's Start.
	padding := 50 * time.Millisecond
	res := &domain.Result{
		Words: []domain.Word{
			{Text: "one", Start: 200 * time.Millisecond, End: 400 * time.Millisecond},
			{Text: "two", Start: 600 * time.Millisecond, End: 800 * time.Millisecond},
			{Text: "three", Start: 1000 * time.Millisecond, End: 1200 * time.Millisecond},
		},
	}
	applyDavinci(res, &domain.DaVinciOptions{
		SilentPortionThreshold: 5000 * time.Millisecond, // high threshold → no pause markers
		PaddingStart:           padding,
	})
	require.Equal(t, 150*time.Millisecond, res.Words[0].Start, "word 0: full padding subtracted")
	require.Equal(t, 550*time.Millisecond, res.Words[1].Start, "word 1: full padding subtracted")
	require.Equal(t, 950*time.Millisecond, res.Words[2].Start, "word 2: full padding subtracted")
}

func TestDavinciPadding_CapsAtHalfPreviousGap(t *testing.T) {
	// Gap between word 0 end (300ms) and word 1 start (360ms) = 60ms.
	// half-gap = 30ms; padding = 50ms → cap applies → subtract only 30ms.
	padding := 50 * time.Millisecond
	res := &domain.Result{
		Words: []domain.Word{
			{Text: "one", Start: 100 * time.Millisecond, End: 300 * time.Millisecond},
			{Text: "two", Start: 360 * time.Millisecond, End: 600 * time.Millisecond},
		},
	}
	applyDavinci(res, &domain.DaVinciOptions{
		SilentPortionThreshold: 5000 * time.Millisecond,
		PaddingStart:           padding,
	})
	// word 0: first word, gap is infinite → full padding (100ms - 50ms = 50ms)
	require.Equal(t, 50*time.Millisecond, res.Words[0].Start, "word 0: full padding")
	// word 1: gap = 360 - 300 = 60ms; half-gap = 30ms < 50ms → subtract 30ms
	require.Equal(t, 330*time.Millisecond, res.Words[1].Start, "word 1: capped at half-gap")
}

func TestDavinciPadding_FirstWordCanShiftIntoNegative(t *testing.T) {
	// First word starts at 30ms, padding is 50ms → would go to -20ms → clamp to 0.
	padding := 50 * time.Millisecond
	res := &domain.Result{
		Words: []domain.Word{
			{Text: "hello", Start: 30 * time.Millisecond, End: 300 * time.Millisecond},
		},
	}
	applyDavinci(res, &domain.DaVinciOptions{
		SilentPortionThreshold: 5000 * time.Millisecond,
		PaddingStart:           padding,
	})
	require.Equal(t, time.Duration(0), res.Words[0].Start, "first word clamped to zero")
}

func TestDavinciPadding_ZeroIsNoOp(t *testing.T) {
	res := &domain.Result{
		Words: []domain.Word{
			{Text: "one", Start: 200 * time.Millisecond, End: 400 * time.Millisecond},
			{Text: "two", Start: 600 * time.Millisecond, End: 800 * time.Millisecond},
		},
	}
	applyDavinci(res, &domain.DaVinciOptions{
		SilentPortionThreshold: 5000 * time.Millisecond,
		PaddingStart:           0,
	})
	require.Equal(t, 200*time.Millisecond, res.Words[0].Start)
	require.Equal(t, 600*time.Millisecond, res.Words[1].Start)
}

// --padding-end tests mirror --padding-start semantics but shrink End instead.

func TestDavinciPaddingEnd_SubtractsFromEnd(t *testing.T) {
	// Words separated by 200ms gaps; padding-end is 50ms → half-gap = 100ms > 50ms,
	// so the full 50ms is subtracted from every word's End.
	res := &domain.Result{
		Words: []domain.Word{
			{Text: "one", Start: 200 * time.Millisecond, End: 400 * time.Millisecond},
			{Text: "two", Start: 600 * time.Millisecond, End: 800 * time.Millisecond},
			{Text: "three", Start: 1000 * time.Millisecond, End: 1200 * time.Millisecond},
		},
	}
	applyDavinci(res, &domain.DaVinciOptions{
		SilentPortionThreshold: 5000 * time.Millisecond,
		PaddingEnd:             50 * time.Millisecond,
	})
	require.Equal(t, 350*time.Millisecond, res.Words[0].End, "word 0: full padding subtracted from End")
	require.Equal(t, 750*time.Millisecond, res.Words[1].End, "word 1: full padding subtracted from End")
	// Last word: no next word → full PaddingEnd subtracted, clamped to >= Start.
	require.Equal(t, 1150*time.Millisecond, res.Words[2].End, "word 2 (last): full padding subtracted from End")
}

func TestDavinciPaddingEnd_CapsAtHalfNextGap(t *testing.T) {
	// Gap between word 0 end (300ms) and word 1 start (360ms) = 60ms.
	// half-gap = 30ms; padding-end = 50ms → cap applies → subtract only 30ms.
	res := &domain.Result{
		Words: []domain.Word{
			{Text: "one", Start: 100 * time.Millisecond, End: 300 * time.Millisecond},
			{Text: "two", Start: 360 * time.Millisecond, End: 600 * time.Millisecond},
		},
	}
	applyDavinci(res, &domain.DaVinciOptions{
		SilentPortionThreshold: 5000 * time.Millisecond,
		PaddingEnd:             50 * time.Millisecond,
	})
	// word 0: gap to next = 60ms; half-gap = 30ms < 50ms → subtract 30ms
	require.Equal(t, 270*time.Millisecond, res.Words[0].End, "word 0: capped at half-gap")
	// word 1 (last): no next word → full padding (600ms - 50ms = 550ms)
	require.Equal(t, 550*time.Millisecond, res.Words[1].End, "word 1 (last): full padding subtracted")
}

func TestDavinciPaddingEnd_NeverCrossesStart(t *testing.T) {
	// Word with End barely above Start; large PaddingEnd must be clamped to Start.
	res := &domain.Result{
		Words: []domain.Word{
			{Text: "brief", Start: 500 * time.Millisecond, End: 510 * time.Millisecond},
		},
	}
	applyDavinci(res, &domain.DaVinciOptions{
		SilentPortionThreshold: 5000 * time.Millisecond,
		PaddingEnd:             200 * time.Millisecond,
	})
	require.Equal(t, res.Words[0].Start, res.Words[0].End, "End must clamp to Start, not go below it")
}

func TestDavinciPaddingEnd_ZeroIsNoOp(t *testing.T) {
	res := &domain.Result{
		Words: []domain.Word{
			{Text: "one", Start: 200 * time.Millisecond, End: 400 * time.Millisecond},
			{Text: "two", Start: 600 * time.Millisecond, End: 800 * time.Millisecond},
		},
	}
	applyDavinci(res, &domain.DaVinciOptions{
		SilentPortionThreshold: 5000 * time.Millisecond,
		PaddingEnd:             0,
	})
	require.Equal(t, 400*time.Millisecond, res.Words[0].End)
	require.Equal(t, 800*time.Millisecond, res.Words[1].End)
}

// TestDavinciPaddingEnd_AppliesFullyEvenWhenPauseFollows verifies that
// PaddingEnd is applied in full when the next real word is far away (even
// though that gap would trigger a pause marker).  Before the order fix, the
// synthetic "(...)" marker sat between the words, making the gap appear as 0
// and capping the shrink to 0ms — the real word's End was never adjusted.
func TestDavinciPaddingEnd_AppliesFullyEvenWhenPauseFollows(t *testing.T) {
	// Gap between word 0 End (500ms) and word 1 Start (3000ms) = 2500ms.
	// That gap exceeds the 1500ms threshold, so a pause marker will be
	// inserted — but padding must be applied first, before the marker exists.
	// half-gap = 1250ms >> PaddingEnd (50ms), so the full 50ms must be applied.
	res := &domain.Result{
		Words: []domain.Word{
			{Text: "hello", Start: 0, End: 500 * time.Millisecond},
			{Text: "world", Start: 3000 * time.Millisecond, End: 3500 * time.Millisecond},
		},
	}
	applyDavinci(res, &domain.DaVinciOptions{
		SilentPortionThreshold: 1500 * time.Millisecond,
		PaddingEnd:             50 * time.Millisecond,
	})
	// Find "hello" in the output (pause marker was inserted; it is not "hello").
	var helloEnd time.Duration = -1
	for _, w := range res.Words {
		if w.Text == "hello" {
			helloEnd = w.End
		}
	}
	require.NotEqual(t, time.Duration(-1), helloEnd, "word 'hello' must appear in output")
	require.Equal(t, 450*time.Millisecond, helloEnd,
		"PaddingEnd must apply the full 50ms even when a pause marker follows")
}

// TestDavinciPaddingStart_AppliesFullyEvenWhenPausePrecedes verifies that
// PaddingStart is applied in full when the preceding real word is far away
// (even though that gap would trigger a pause marker).
func TestDavinciPaddingStart_AppliesFullyEvenWhenPausePrecedes(t *testing.T) {
	// Gap between word 0 End (500ms) and word 1 Start (3000ms) = 2500ms.
	// half-gap = 1250ms >> PaddingStart (50ms), so the full 50ms must be applied.
	res := &domain.Result{
		Words: []domain.Word{
			{Text: "hello", Start: 0, End: 500 * time.Millisecond},
			{Text: "world", Start: 3000 * time.Millisecond, End: 3500 * time.Millisecond},
		},
	}
	applyDavinci(res, &domain.DaVinciOptions{
		SilentPortionThreshold: 1500 * time.Millisecond,
		PaddingStart:           50 * time.Millisecond,
	})
	// Find "world" in the output.
	var worldStart time.Duration = -1
	for _, w := range res.Words {
		if w.Text == "world" {
			worldStart = w.Start
		}
	}
	require.NotEqual(t, time.Duration(-1), worldStart, "word 'world' must appear in output")
	require.Equal(t, 2950*time.Millisecond, worldStart,
		"PaddingStart must apply the full 50ms even when a pause marker precedes")
}
