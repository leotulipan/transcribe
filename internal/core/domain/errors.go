package domain

import (
	"errors"
	"fmt"
)

var (
	ErrConfigMissing   = errors.New("config error")
	ErrProviderMissing = errors.New("provider not configured")
	ErrFFmpegMissing   = errors.New("ffmpeg not found")
	ErrCanceled        = errors.New("canceled")
)

// ErrIncompatible signals a request/model/format mismatch caught before any
// expensive work runs.
type ErrIncompatible struct {
	Provider ProviderID
	Model    string
	Format   OutputFormat
	Reason   string
}

func (e ErrIncompatible) Error() string {
	return fmt.Sprintf("incompatible: %s/%s cannot produce %s — %s",
		e.Provider, e.Model, e.Format, e.Reason)
}

// ErrProvider wraps an upstream API failure with classification hints.
type ErrProvider struct {
	Provider   ProviderID
	StatusCode int
	Retryable  bool
	Cause      error
}

func (e *ErrProvider) Error() string {
	if e.StatusCode == 0 {
		return fmt.Sprintf("%s: %v", e.Provider, e.Cause)
	}
	return fmt.Sprintf("%s: http %d: %v", e.Provider, e.StatusCode, e.Cause)
}

func (e *ErrProvider) Unwrap() error { return e.Cause }
