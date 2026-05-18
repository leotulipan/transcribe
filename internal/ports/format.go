package ports

import "github.com/leotulipan/transcribe/internal/core/domain"

type FormatWriter interface {
    Format() domain.OutputFormat
    Write(r *domain.Result, dst string) error
}
