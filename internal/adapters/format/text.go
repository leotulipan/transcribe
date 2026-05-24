package format

import (
	"os"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

type Text struct{}

func NewText() *Text { return &Text{} }

func (Text) Format() domain.OutputFormat { return domain.FormatText }

func (Text) Write(r *domain.Result, dst string, _ domain.WriteOpts) error {
	return os.WriteFile(dst, []byte(r.Text), 0o644)
}
