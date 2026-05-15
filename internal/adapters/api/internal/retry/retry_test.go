package retry

import (
	"context"
	"errors"
	"net"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

type fakeNetErr struct{}

func (fakeNetErr) Error() string   { return "fake net timeout" }
func (fakeNetErr) Timeout() bool   { return true }
func (fakeNetErr) Temporary() bool { return true }

var _ net.Error = fakeNetErr{}

func TestDo_RetriesTransientAndSucceeds(t *testing.T) {
	var calls int
	err := Do(context.Background(), 3, 1*time.Millisecond, func() error {
		calls++
		if calls < 3 {
			return fakeNetErr{}
		}
		return nil
	})
	require.NoError(t, err)
	require.Equal(t, 3, calls)
}

func TestDo_DoesNotRetryPermanent(t *testing.T) {
	var calls int
	boom := errors.New("auth failure")
	err := Do(context.Background(), 5, 1*time.Millisecond, func() error {
		calls++
		return boom
	})
	require.ErrorIs(t, err, boom)
	require.Equal(t, 1, calls)
}

func TestDo_GivesUpAfterMaxAttempts(t *testing.T) {
	var calls int
	err := Do(context.Background(), 2, 1*time.Millisecond, func() error {
		calls++
		return fakeNetErr{}
	})
	require.Error(t, err)
	require.Equal(t, 2, calls)
}

func TestIsRetryable_HTTPStatus(t *testing.T) {
	require.True(t, IsRetryable(HTTPError{StatusCode: 500}))
	require.True(t, IsRetryable(HTTPError{StatusCode: 429}))
	require.True(t, IsRetryable(HTTPError{StatusCode: 503}))
	require.False(t, IsRetryable(HTTPError{StatusCode: 400}))
	require.False(t, IsRetryable(HTTPError{StatusCode: 401}))
}
