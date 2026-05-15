package retry

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"net"
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
