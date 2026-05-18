package cli

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

func TestExitCodeFor(t *testing.T) {
	require.Equal(t, 0, ExitCodeFor(nil))
	require.Equal(t, 3, ExitCodeFor(domain.ErrFFmpegMissing))
	require.Equal(t, 3, ExitCodeFor(domain.ErrConfigMissing))
	require.Equal(t, 3, ExitCodeFor(domain.ErrProviderMissing))
	require.Equal(t, 2, ExitCodeFor(domain.ErrIncompatible{}))
	require.Equal(t, 4, ExitCodeFor(&domain.ErrProvider{}))
	require.Equal(t, 130, ExitCodeFor(domain.ErrCanceled))
	require.Equal(t, 1, ExitCodeFor(errors.New("unexpected")))
}
