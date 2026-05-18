package services

import (
	"context"
	"errors"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

// transient reports whether an error is the kind that a future retry might
// succeed against — drives the keep-intermediate cleanup policy.
func transient(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, domain.ErrCanceled) {
		return false
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return true
	}
	var pe *domain.ErrProvider
	if errors.As(err, &pe) {
		return pe.Retryable
	}
	return false
}
