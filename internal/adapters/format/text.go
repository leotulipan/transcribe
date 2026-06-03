package format

import (
	"os"
	"strings"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

type Text struct{}

func NewText() *Text { return &Text{} }

func (Text) Format() domain.OutputFormat { return domain.FormatText }

func (Text) Write(r *domain.Result, dst string, opts domain.WriteOpts) error {
	if opts.SpeakerLabels && len(r.Words) > 0 {
		return os.WriteFile(dst, []byte(renderSpeakerText(r.Words)), 0o644)
	}
	return os.WriteFile(dst, []byte(r.Text), 0o644)
}

// renderSpeakerText groups consecutive words by speaker into one paragraph per
// turn, each prefixed with the speaker label. A new paragraph starts whenever the
// speaker changes.
func renderSpeakerText(words []domain.Word) string {
	var b strings.Builder
	var cur string
	var line []string
	flush := func() {
		if len(line) == 0 {
			return
		}
		b.WriteString(speakerPrefix(cur))
		b.WriteString(strings.Join(line, " "))
		b.WriteString("\n\n")
		line = nil
	}
	for _, w := range words {
		if w.Speaker != cur && len(line) > 0 {
			flush()
		}
		cur = w.Speaker
		line = append(line, w.Text)
	}
	flush()
	return b.String()
}
