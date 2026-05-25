package services

import (
	"strings"
	"time"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

// snapToFrames snaps every word's Start and End to the nearest frame boundary
// at the given fps, then applies the per-boundary frame offsets. Results are
// clamped so Start >= 0 and End >= Start.
func snapToFrames(words []domain.Word, fps float64, offsetStart, offsetEnd int) {
	if fps <= 0 || len(words) == 0 {
		return
	}
	fn := time.Duration(float64(time.Second) / fps)
	if fn <= 0 {
		return
	}
	for i := range words {
		w := &words[i]

		frameStart := int64((float64(w.Start) / float64(fn)) + 0.5)
		frameEnd := int64((float64(w.End) / float64(fn)) + 0.5)

		newStart := time.Duration(frameStart+int64(offsetStart)) * fn
		newEnd := time.Duration(frameEnd+int64(offsetEnd)) * fn

		if newStart < 0 {
			newStart = 0
		}
		if newEnd < newStart {
			newEnd = newStart
		}
		w.Start = newStart
		w.End = newEnd
	}
}

// applyDavinci mutates result.Words in place: pause markers get inserted as
// synthetic words with text "(...)", and filler-word matches get uppercased
// so the DaVinci format writer renders them on their own line.
//
// Processing order:
//  1. PaddingStart — shift real-word Start times earlier (based on real-word gaps).
//  2. PaddingEnd   — shrink real-word End times (based on real-word gaps).
//  3. Frame-snap   — when FPS > 0, quantize Start/End to the frame grid and apply offsets.
//  4. Filler + pause-marker pass — insert synthetic "(...)" words and uppercase
//     fillers. Pause markers are based on the padding-adjusted gaps, which is
//     architecturally correct: the silence between words is measured after any
//     timestamp adjustments have been applied.
func applyDavinci(r *domain.Result, opts *domain.DaVinciOptions) {
	if opts == nil || len(r.Words) == 0 {
		return
	}
	fillers := opts.FillerWords
	if len(fillers) == 0 {
		fillers = domain.DefaultFillerWords
	}
	threshold := opts.SilentPortionThreshold
	if threshold <= 0 {
		threshold = 1500 * time.Millisecond
	}

	fillerSet := map[string]struct{}{}
	for _, f := range fillers {
		fillerSet[strings.ToLower(f)] = struct{}{}
	}

	// Pass 1: PaddingStart — operate on real words only, no insertions yet.
	if opts.PaddingStart > 0 {
		var prevWordEnd time.Duration = -1
		for i := range r.Words {
			w := &r.Words[i]
			shift := opts.PaddingStart
			if prevWordEnd >= 0 {
				gap := w.Start - prevWordEnd
				halfGap := gap / 2
				if halfGap < shift {
					shift = halfGap
				}
			}
			w.Start -= shift
			if w.Start < 0 {
				w.Start = 0
			}
			prevWordEnd = r.Words[i].End
		}
	}

	// Pass 2: PaddingEnd — operate on real words only, no insertions yet.
	if opts.PaddingEnd > 0 {
		for i := range r.Words {
			w := &r.Words[i]
			shrink := opts.PaddingEnd
			if i+1 < len(r.Words) {
				gap := r.Words[i+1].Start - w.End
				halfGap := gap / 2
				if halfGap < shrink {
					shrink = halfGap
				}
			}
			w.End -= shrink
			if w.End < w.Start {
				w.End = w.Start
			}
		}
	}

	// Pass 3: Frame-snap — quantize to the frame grid after padding, before pause markers.
	snapToFrames(r.Words, opts.FPS, opts.FPSOffsetStart, opts.FPSOffsetEnd)

	// Pass 4: Filler handling + pause-marker insertion.
	// Gaps are measured on the padding- and frame-snap-adjusted timestamps.
	var out []domain.Word
	prevEnd := time.Duration(-1)
	for i, w := range r.Words {
		if i > 0 && threshold > 0 && !opts.SuppressPauses {
			gap := w.Start - prevEnd
			if gap >= threshold {
				out = append(out, domain.Word{
					Text:  "(...)",
					Start: prevEnd,
					End:   w.Start,
				})
			}
		}
		text := w.Text
		if _, ok := fillerSet[strings.ToLower(strings.TrimFunc(text, isPunct))]; ok {
			if opts.RemoveFillers {
				prevEnd = w.End
				continue
			}
			if !opts.SuppressFillerLines {
				text = strings.ToUpper(text)
			}
		}
		out = append(out, domain.Word{
			Text:       text,
			Start:      w.Start,
			End:        w.End,
			Confidence: w.Confidence,
		})
		prevEnd = w.End
	}
	r.Words = out
}

func isPunct(r rune) bool {
	switch r {
	case '.', ',', '!', '?', ';', ':':
		return true
	}
	return false
}
