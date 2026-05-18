package services

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

func TestJob_LifecycleSuccess(t *testing.T) {
	j := newJob(context.Background(), domain.Request{}, "id-1")
	var ran atomic.Bool
	go j.run(func(emit func(domain.ProgressEvent)) (*domain.Result, error) {
		emit(domain.ProgressEvent{Stage: domain.StageProbing})
		ran.Store(true)
		return &domain.Result{Text: "done"}, nil
	})

	var seen []domain.Stage
	for ev := range j.Progress() {
		seen = append(seen, ev.Stage)
	}
	res, err := j.Wait()
	require.NoError(t, err)
	require.Equal(t, "done", res.Text)
	require.True(t, ran.Load())
	require.Contains(t, seen, domain.StageProbing)
}

func TestJob_CancelStopsWaitingFn(t *testing.T) {
	j := newJob(context.Background(), domain.Request{}, "id-2")
	started := make(chan struct{})
	go j.run(func(emit func(domain.ProgressEvent)) (*domain.Result, error) {
		close(started)
		<-j.ctx.Done()
		return nil, j.ctx.Err()
	})
	<-started
	j.Cancel()
	_, err := j.Wait()
	require.ErrorIs(t, err, context.Canceled)
}

func TestJob_WaitIsRepeatable(t *testing.T) {
	j := newJob(context.Background(), domain.Request{}, "id-3")
	boom := errors.New("boom")
	go j.run(func(emit func(domain.ProgressEvent)) (*domain.Result, error) {
		return nil, boom
	})
	_, err1 := j.Wait()
	require.ErrorIs(t, err1, boom)
	_, err2 := j.Wait()
	require.ErrorIs(t, err2, boom)
}

func TestJob_ProgressClosedAfterDone(t *testing.T) {
	j := newJob(context.Background(), domain.Request{}, "id-4")
	go j.run(func(emit func(domain.ProgressEvent)) (*domain.Result, error) {
		return &domain.Result{}, nil
	})
	// Drain
	for range j.Progress() {
	}
	select {
	case <-j.Progress():
		// already closed; receive on closed chan returns immediately — OK
	case <-time.After(100 * time.Millisecond):
		t.Fatal("Progress channel should be closed after job ends")
	}
}
