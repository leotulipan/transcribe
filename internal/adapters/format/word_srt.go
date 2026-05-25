package format

import (
	"os"
	"strings"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

type WordSRT struct{}

func NewWordSRT() *WordSRT { return &WordSRT{} }

func (WordSRT) Format() domain.OutputFormat { return domain.FormatWordSRT }

func (WordSRT) Write(r *domain.Result, dst string, opts domain.WriteOpts) error {
	var b strings.Builder
	for i, w := range r.Words {
		b.WriteString(itoa(i + 1))
		b.WriteByte('\n')
		b.WriteString(formatTimecodeOffset(w.Start, opts.StartHour))
		b.WriteString(" --> ")
		b.WriteString(formatTimecodeOffset(w.End, opts.StartHour))
		b.WriteByte('\n')
		b.WriteString(w.Text)
		b.WriteString("\n\n")
	}
	return os.WriteFile(dst, []byte(b.String()), 0o644)
}
