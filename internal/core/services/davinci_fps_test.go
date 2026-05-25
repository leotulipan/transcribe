package services

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

// frameNs returns the duration of one frame at the given fps, matching snapToFrames.
func frameNs(fps float64) time.Duration {
	return time.Duration(float64(time.Second) / fps)
}

func TestDavinciFPS_ZeroFPSIsNoOp(t *testing.T) {
	// fps=0 means no snapping; word times must come through unchanged.
	res := &domain.Result{
		Words: []domain.Word{
			{Text: "one", Start: 50 * time.Millisecond, End: 300 * time.Millisecond},
		},
	}
	applyDavinci(res, &domain.DaVinciOptions{
		SilentPortionThreshold: 5000 * time.Millisecond,
		FPS:                    0,
		FPSOffsetStart:         -1,
		FPSOffsetEnd:           0,
	})
	require.Equal(t, 50*time.Millisecond, res.Words[0].Start)
	require.Equal(t, 300*time.Millisecond, res.Words[0].End)
}

func TestDavinciFPS_SnapsToNearestFrame_24fps(t *testing.T) {
	// At 24fps, frame duration = 41,666,666ns (~41.667ms).
	// A word starting at 50ms is closest to frame 1 (41.667ms).
	// offsetStart=0 → snapped Start = frame1 = 41.667ms (the frame ns value).
	// offsetStart=-1 (default) → snapped Start = frame0 = 0ms.
	fn := frameNs(24)

	// offsetStart=0: snap 50ms → frame 1
	res := &domain.Result{
		Words: []domain.Word{
			{Text: "a", Start: 50 * time.Millisecond, End: 500 * time.Millisecond},
		},
	}
	applyDavinci(res, &domain.DaVinciOptions{
		SilentPortionThreshold: 5000 * time.Millisecond,
		FPS:                    24,
		FPSOffsetStart:         0,
		FPSOffsetEnd:           0,
	})
	require.Equal(t, fn, res.Words[0].Start, "50ms snaps to frame 1 at 24fps with offset 0")

	// offsetStart=-1: snap 50ms → frame 1, then subtract 1 → frame 0 = 0
	res2 := &domain.Result{
		Words: []domain.Word{
			{Text: "a", Start: 50 * time.Millisecond, End: 500 * time.Millisecond},
		},
	}
	applyDavinci(res2, &domain.DaVinciOptions{
		SilentPortionThreshold: 5000 * time.Millisecond,
		FPS:                    24,
		FPSOffsetStart:         -1,
		FPSOffsetEnd:           0,
	})
	require.Equal(t, time.Duration(0), res2.Words[0].Start, "50ms snaps to frame 0 with offsetStart=-1")
}

func TestDavinciFPS_SnapsEndToNearestFrame(t *testing.T) {
	// At 24fps, a word ending at 70ms: nearest frame = round(70ms / 41.667ms) = round(1.68) = 2 → frame 2 = 83.333ms.
	// offsetEnd=0 → End = frame2 * frameNs.
	fn := frameNs(24)
	res := &domain.Result{
		Words: []domain.Word{
			{Text: "a", Start: 0, End: 70 * time.Millisecond},
		},
	}
	applyDavinci(res, &domain.DaVinciOptions{
		SilentPortionThreshold: 5000 * time.Millisecond,
		FPS:                    24,
		FPSOffsetStart:         0,
		FPSOffsetEnd:           0,
	})
	require.Equal(t, 2*fn, res.Words[0].End, "70ms snaps to frame 2 at 24fps")
}

func TestDavinciFPS_NegativeOffsetClampsAtZero(t *testing.T) {
	// A word starting at 20ms at 24fps: nearest frame = round(20ms / 41.667ms) = round(0.48) = 0.
	// With offsetStart=-1 → frame -1 → Start = -1 * frameNs → clamp to 0.
	res := &domain.Result{
		Words: []domain.Word{
			{Text: "a", Start: 20 * time.Millisecond, End: 500 * time.Millisecond},
		},
	}
	applyDavinci(res, &domain.DaVinciOptions{
		SilentPortionThreshold: 5000 * time.Millisecond,
		FPS:                    24,
		FPSOffsetStart:         -1,
		FPSOffsetEnd:           0,
	})
	require.Equal(t, time.Duration(0), res.Words[0].Start, "negative offset at frame 0 must clamp to 0")
}

func TestDavinciFPS_EndCanNotPrecedeStart(t *testing.T) {
	// Word from 0 to 10ms at 24fps.
	// End=10ms → nearest frame = round(10ms / 41.667ms) = round(0.24) = 0.
	// With offsetEnd=-1 → frame -1 → End = -1 * frameNs → clamp to Start.
	res := &domain.Result{
		Words: []domain.Word{
			{Text: "a", Start: 0, End: 10 * time.Millisecond},
		},
	}
	applyDavinci(res, &domain.DaVinciOptions{
		SilentPortionThreshold: 5000 * time.Millisecond,
		FPS:                    24,
		FPSOffsetStart:         0,
		FPSOffsetEnd:           -1,
	})
	// Start snaps to frame 0 = 0; End snaps to frame 0 then -1 → clamp to Start = 0.
	require.GreaterOrEqual(t, int64(res.Words[0].End), int64(res.Words[0].Start),
		"End must never precede Start after offset application")
}

func TestDavinciFPS_30fps(t *testing.T) {
	// At 30fps, frame duration = 33,333,333ns (~33.333ms).
	// Word starting at 50ms → nearest frame = round(50ms / 33.333ms) = round(1.5) = 2 → frame 2 = 66.667ms.
	// With offsetStart=0.
	fn := frameNs(30)
	res := &domain.Result{
		Words: []domain.Word{
			{Text: "a", Start: 50 * time.Millisecond, End: 200 * time.Millisecond},
		},
	}
	applyDavinci(res, &domain.DaVinciOptions{
		SilentPortionThreshold: 5000 * time.Millisecond,
		FPS:                    30,
		FPSOffsetStart:         0,
		FPSOffsetEnd:           0,
	})
	require.Equal(t, 2*fn, res.Words[0].Start, "50ms snaps to frame 2 at 30fps")
}

func TestDavinciFPS_IntegratesAfterPadding(t *testing.T) {
	// Padding shifts Start earlier, then frame-snap quantizes to the frame grid.
	// Word at Start=200ms, PaddingStart=50ms → adjusted Start = 150ms.
	// At 24fps, frame = 41.667ms. Nearest frame to 150ms = round(150/41.667) = round(3.6) = 4 → 166.667ms.
	// With FPSOffsetStart=0, final Start must be exactly 4 * frameNs(24) — on a frame boundary.
	fn := frameNs(24)
	res := &domain.Result{
		Words: []domain.Word{
			// gap before this word is large → full padding applies
			{Text: "a", Start: 200 * time.Millisecond, End: 500 * time.Millisecond},
		},
	}
	applyDavinci(res, &domain.DaVinciOptions{
		SilentPortionThreshold: 5000 * time.Millisecond,
		PaddingStart:           50 * time.Millisecond,
		FPS:                    24,
		FPSOffsetStart:         0,
		FPSOffsetEnd:           0,
	})
	got := res.Words[0].Start
	// Verify the result is an exact multiple of frameNs (i.e., on the frame grid).
	require.Equal(t, time.Duration(0), got%fn,
		"after padding+snap, Start must land on a frame boundary")
	// Also verify it is the frame nearest to 150ms (post-padding), which is frame 4.
	require.Equal(t, 4*fn, got, "Start snaps to frame 4 (nearest to 150ms at 24fps)")
}
