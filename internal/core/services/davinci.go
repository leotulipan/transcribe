package services

import (
	"strings"
	"time"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

// applyDavinci mutates result.Words in place: pause markers get inserted as
// synthetic words with text "(...)", and filler-word matches get uppercased
// so the DaVinci format writer renders them on their own line.
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

	if opts.PaddingStart > 0 {
		var prevWordEnd time.Duration = -1
		for i := range r.Words {
			// Pause markers (text "(...)") sit in r.Words with gap == 0 to the preceding word,
			// so the cap rule means adjacent words receive zero shift — intentional, matches Python.
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
}

func isPunct(r rune) bool {
	switch r {
	case '.', ',', '!', '?', ';', ':':
		return true
	}
	return false
}
