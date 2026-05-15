package format

import (
	"time"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

// block is one subtitle entry: a contiguous run of words sharing a start/end.
type block struct {
	Words []domain.Word
	Start time.Duration
	End   time.Duration
}

// groupWords folds a flat word list into subtitle blocks. A new block starts
// when the running block already has maxWords entries OR the gap between the
// previous word's End and the next word's Start exceeds maxGap.
func groupWords(words []domain.Word, maxWords int, maxGap time.Duration) []block {
	if len(words) == 0 {
		return nil
	}
	var (
		out []block
		cur block
	)
	cur.Words = []domain.Word{words[0]}
	cur.Start = words[0].Start
	cur.End = words[0].End

	for i := 1; i < len(words); i++ {
		w := words[i]
		gap := w.Start - cur.End
		if len(cur.Words) >= maxWords || gap > maxGap {
			out = append(out, cur)
			cur = block{Words: []domain.Word{w}, Start: w.Start, End: w.End}
			continue
		}
		cur.Words = append(cur.Words, w)
		cur.End = w.End
	}
	out = append(out, cur)
	return out
}

// formatTimecode renders an SRT-style timecode "HH:MM:SS,mmm".
func formatTimecode(d time.Duration) string {
	if d < 0 {
		d = 0
	}
	h := d / time.Hour
	d -= h * time.Hour
	m := d / time.Minute
	d -= m * time.Minute
	s := d / time.Second
	d -= s * time.Second
	ms := d / time.Millisecond
	return twoDigit(int(h)) + ":" + twoDigit(int(m)) + ":" + twoDigit(int(s)) + "," + threeDigit(int(ms))
}

func twoDigit(n int) string {
	if n < 10 {
		return "0" + itoa(n)
	}
	return itoa(n)
}

func threeDigit(n int) string {
	switch {
	case n < 10:
		return "00" + itoa(n)
	case n < 100:
		return "0" + itoa(n)
	default:
		return itoa(n)
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}
