package format

import (
	"os"
	"strings"
	"time"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

const (
	srtMaxWordsPerBlock = 7
	srtMaxGap           = 3 * time.Second
)

type SRT struct{}

func NewSRT() *SRT { return &SRT{} }

func (SRT) Format() domain.OutputFormat { return domain.FormatSRT }

func (SRT) Write(r *domain.Result, dst string, opts domain.WriteOpts) error {
	blocks := groupWords(r.Words, srtMaxWordsPerBlock, srtMaxGap)
	var b strings.Builder
	for i, blk := range blocks {
		b.WriteString(itoa(i + 1))
		b.WriteByte('\n')
		b.WriteString(formatTimecode(blk.Start))
		b.WriteString(" --> ")
		b.WriteString(formatTimecode(blk.End))
		b.WriteByte('\n')
		lines := wrapByChars(blk.Words, opts.MaxCharsPerLine)
		for li, line := range lines {
			if li > 0 {
				b.WriteByte('\n')
			}
			for j, w := range line {
				if j > 0 {
					b.WriteByte(' ')
				}
				b.WriteString(w.Text)
			}
		}
		b.WriteString("\n\n")
	}
	return os.WriteFile(dst, []byte(b.String()), 0o644)
}
