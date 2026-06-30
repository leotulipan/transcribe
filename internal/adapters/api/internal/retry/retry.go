package retry

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"net"
	"strings"
	"time"
)

// HTTPError is a lightweight wrapper API adapters can return so the retry
// helper can classify by status code.
type HTTPError struct {
	StatusCode int
	Message    string
}

func (e HTTPError) Error() string {
	if e.Message != "" {
		return fmt.Sprintf("http %d: %s", e.StatusCode, e.Message)
	}
	return fmt.Sprintf("http %d", e.StatusCode)
}

// IsRetryable classifies common transient failures.
func IsRetryable(err error) bool {
	if err == nil {
		return false
	}
	var he HTTPError
	if errors.As(err, &he) {
		return he.StatusCode == 429 || (he.StatusCode >= 500 && he.StatusCode < 600)
	}
	var ne net.Error
	if errors.As(err, &ne) && ne.Timeout() {
		return true
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return true
	}
	return isTransientConnError(err)
}

// transientConnMarkers are substrings of connection-level failures where the
// server dropped the connection before sending a response, so retrying on a
// fresh connection is safe. The HTTP/2 GOAWAY error returned by net/http's
// bundled http2 transport (`http2GoAwayError`) is unexported and is not a
// net.Error, so we have to match it by message. "server_shutting_down" is the
// debug payload providers such as ElevenLabs send while a node is recycled.
var transientConnMarkers = []string{
	"GOAWAY",
	"server_shutting_down",
	"connection reset by peer",
	"broken pipe",
	"unexpected EOF",
}

// isTransientConnError reports whether err is a recoverable connection-level
// failure that warrants a retry. It matches on the error message because the
// underlying types (e.g. the bundled http2 GoAwayError) are unexported.
func isTransientConnError(err error) bool {
	msg := err.Error()
	for _, marker := range transientConnMarkers {
		if strings.Contains(msg, marker) {
			return true
		}
	}
	return false
}

// Do executes fn up to attempts times, returning the final error.
// Backoff: base * 2^(i-1) + jitter ∈ [0, base].
func Do(ctx context.Context, attempts int, base time.Duration, fn func() error) error {
	if attempts < 1 {
		attempts = 1
	}
	var err error
	for i := 1; i <= attempts; i++ {
		err = fn()
		if err == nil {
			return nil
		}
		if !IsRetryable(err) || i == attempts {
			return err
		}
		wait := base*time.Duration(1<<(i-1)) + time.Duration(rand.Int63n(int64(base)))
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(wait):
		}
	}
	return err
}
