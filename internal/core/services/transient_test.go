package services

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

func TestTransient_Classification(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want bool
	}{
		{"nil", nil, false},
		{"canceled", context.Canceled, false},
		{"deadline exceeded", context.DeadlineExceeded, true},
		{"ErrCanceled", domain.ErrCanceled, false},
		{"ErrIncompatible", domain.ErrIncompatible{Reason: "x"}, false},
		{"provider retryable", &domain.ErrProvider{Retryable: true, Cause: errors.New("503")}, true},
		{"provider permanent", &domain.ErrProvider{Retryable: false, Cause: errors.New("401")}, false},
		{"unknown", errors.New("other"), false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			require.Equal(t, c.want, transient(c.err))
		})
	}
}
