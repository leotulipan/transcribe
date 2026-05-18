package services

import "testing"

// --padding-start subtracts a small amount from each word's start time so the
// DaVinci-imported subtitle leads its audio (compensates editor preview lag).
// The DaVinciOptions.PaddingStart field exists at internal/core/domain/transcription.go,
// but applyDavinci does not consume it yet.
// See docs/plans/2-feature-parity-completion.md Phase 1b.
//
// Python source: audio_transcribe/transcribe_helpers/output_formatters.py — apply_intelligent_padding
// Python tests:  tests/unit/test_text_processing.py — padding cases

func TestDavinciPadding_SubtractsFromStart(t *testing.T) {
	t.Skip("pending: padding-start logic not wired — see Phase 1b")
	// expected behavior:
	//   given Words with gaps >= 2*padding, every word's Start is shifted
	//   earlier by exactly opts.PaddingStart.
}

func TestDavinciPadding_CapsAtHalfPreviousGap(t *testing.T) {
	t.Skip("pending: padding-start logic not wired — see Phase 1b")
	// expected behavior:
	//   if gap to previous word is < 2*padding, only gap/2 is subtracted
	//   (so the new Start never overlaps the previous End).
}

func TestDavinciPadding_FirstWordCanShiftIntoNegative(t *testing.T) {
	t.Skip("pending: padding-start logic not wired — see Phase 1b")
	// expected behavior:
	//   the very first word has no previous gap; either it gets the full
	//   padding subtracted (clamped to >= 0) or it stays put. Decide and
	//   lock the policy. Python clamps to zero — match that.
}

func TestDavinciPadding_ZeroIsNoOp(t *testing.T) {
	t.Skip("pending: padding-start logic not wired — see Phase 1b")
	// expected behavior:
	//   PaddingStart == 0 → Word.Start values unchanged.
}
