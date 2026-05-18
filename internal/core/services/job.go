package services

import (
	"context"
	"sync"
	"time"

	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

const progressBuffer = 32

type job struct {
	id       string
	req      domain.Request
	progress chan domain.ProgressEvent
	done     chan struct{}
	once     sync.Once
	cancelF  context.CancelFunc
	ctx      context.Context
	started  time.Time

	mu     sync.RWMutex
	result *domain.Result
	err    error
}

func newJob(parent context.Context, req domain.Request, id string) *job {
	ctx, cancel := context.WithCancel(parent)
	return &job{
		id:       id,
		req:      req,
		progress: make(chan domain.ProgressEvent, progressBuffer),
		done:     make(chan struct{}),
		ctx:      ctx,
		cancelF:  cancel,
		started:  time.Now(),
	}
}

var _ ports.Job = (*job)(nil)

func (j *job) ID() string                            { return j.id }
func (j *job) Progress() <-chan domain.ProgressEvent { return j.progress }
func (j *job) Cancel() {
	j.once.Do(func() { j.cancelF() })
}

func (j *job) Wait() (*domain.Result, error) {
	<-j.done
	j.mu.RLock()
	defer j.mu.RUnlock()
	return j.result, j.err
}

// emit pushes a progress event. Drops the event if the buffer is full so a
// slow UI never blocks the pipeline.
func (j *job) emit(ev domain.ProgressEvent) {
	ev.Elapsed = time.Since(j.started)
	select {
	case j.progress <- ev:
	default:
	}
}

// run executes pipeline. The fn receives an emit function and returns the
// final (*Result, error). Always closes the progress channel and the done
// channel on exit.
func (j *job) run(fn func(emit func(domain.ProgressEvent)) (*domain.Result, error)) {
	defer close(j.progress)
	defer close(j.done)

	result, err := fn(j.emit)

	j.mu.Lock()
	j.result = result
	j.err = err
	j.mu.Unlock()
}
