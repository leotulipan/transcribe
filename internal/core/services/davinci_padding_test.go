package services

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

// --padding-start subtracts a small amount from each word's start time so the
// DaVinci-imported subtitle leads its audio (compensates editor preview lag).
// The DaVinciOptions.PaddingStart field exists at internal/core/domain/transcription.go,
// but applyDavinci does not consume it yet.
// See docs/plans/2-feature-parity-completion.md Phase 1b.
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
