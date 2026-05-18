package cli

import (
	"errors"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

const (
	exitOK       = 0
	exitInternal = 1
	exitUsage    = 2
	exitConfig   = 3
	exitProvider = 4
	exitAudio    = 5
	exitCanceled = 130
)

func ExitCodeFor(err error) int {
	if err == nil {
		return exitOK
	}
	if errors.Is(err, domain.ErrFFmpegMissing) {
		return exitConfig
	}
	if errors.Is(err, domain.ErrConfigMissing) {
		return exitConfig
	}
	if errors.Is(err, domain.ErrProviderMissing) {
		return exitConfig
	}
	if errors.Is(err, domain.ErrCanceled) {
		return exitCanceled
	}
	var ei domain.ErrIncompatible
	if errors.As(err, &ei) {
		return exitUsage
	}
	var ep *domain.ErrProvider
	if errors.As(err, &ep) {
		return exitProvider
	}
	return exitInternal
}
