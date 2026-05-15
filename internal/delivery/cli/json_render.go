package cli

import (
	"io"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

// renderJSON is implemented in L4. This stub keeps the build green.
func renderJSON(_ io.Writer, _ interface {
	Progress() <-chan domain.ProgressEvent
	Wait() (*domain.Result, error)
}, _ bool) error {
	return nil
}
