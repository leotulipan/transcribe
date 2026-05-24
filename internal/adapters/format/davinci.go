package format

import (
	"os"
	"strings"
	"time"
	"unicode"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

const (
	davinciMaxWordsPerBlock = 7
	davinciMaxGap           = 3 * time.Second
)

type DaVinci struct{}

func NewDaVinci() *DaVinci { return &DaVinci{} }

func (DaVinci) Format() domain.OutputFormat { return domain.FormatDavinciSRT }

func (DaVinci) Write(r *domain.Result, dst string, opts domain.WriteOpts) error {
	var b strings.Builder
	index := 1

	// Walk words, emitting an SRT block per logical group. Filler-word and
	// pause-marker words break the current group and become their own block.
	var bucket []domain.Word
	flush := func() {
		if len(bucket) == 0 {
			return
		}
		writeBlock(&b, index, bucket, opts.MaxCharsPerLine)
		index++
		bucket = nil
	}
	for _, w := range r.Words {
		switch {
		case w.Text == "(...)":
			flush()
			writeBlock(&b, index, []domain.Word{w}, opts.MaxCharsPerLine)
			index++
		case isAllUpper(w.Text):
			flush()
			writeBlock(&b, index, []domain.Word{w}, opts.MaxCharsPerLine)
			index++
		default:
			// gap-induced break
			if len(bucket) > 0 {
				last := bucket[len(bucket)-1]
				if w.Start-last.End > davinciMaxGap || len(bucket) >= davinciMaxWordsPerBlock {
					flush()
				}
			}
			bucket = append(bucket, w)
		}
	}
	flush()

	return os.WriteFile(dst, []byte(b.String()), 0o644)
}

func writeBlock(b *strings.Builder, idx int, words []domain.Word, maxCharsPerLine int) {
	if len(words) == 0 {
		return
	}
	start := words[0].Start
	end := words[len(words)-1].End
	b.WriteString(itoa(idx))
	b.WriteByte('\n')
	b.WriteString(formatTimecode(start))
	b.WriteString(" --> ")
	b.WriteString(formatTimecode(end))
	b.WriteByte('\n')
	lines := wrapByChars(words, maxCharsPerLine)
	for li, line := range lines {
		if li > 0 {
			b.WriteByte('\n')
		}
		for i, w := range line {
			if i > 0 {
				b.WriteByte(' ')
			}
			b.WriteString(w.Text)
		}
	}
	b.WriteString("\n\n")
}

func isAllUpper(s string) bool {
	if s == "" {
		return false
	}
	hasLetter := false
	for _, r := range s {
		if unicode.IsLetter(r) {
			hasLetter = true
			if !unicode.IsUpper(r) {
				return false
			}
		}
	}
	return hasLetter
}
