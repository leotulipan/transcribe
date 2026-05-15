package domain

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestErrIncompatible_Error(t *testing.T) {
	e := ErrIncompatible{
		Provider: ProviderGroq,
		Model:    "whisper-large-v3",
		Format:   FormatSRT,
		Reason:   "model returns text only",
	}
	msg := e.Error()
	require.Contains(t, msg, "groq")
	require.Contains(t, msg, "whisper-large-v3")
	require.Contains(t, msg, "srt")
	require.Contains(t, msg, "text only")
}

func TestErrProvider_Unwrap(t *testing.T) {
	base := errors.New("boom")
	e := &ErrProvider{Provider: ProviderGroq, StatusCode: 500, Retryable: true, Cause: base}
	require.True(t, errors.Is(e, base))
}
