package gui

import (
	"context"

	"fyne.io/fyne/v2"

	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

// runJob runs a transcription in a goroutine, marshalling progress and final
// result back to the UI thread via fyne.Do.
//
// onProgress is called once per ProgressEvent. onDone is called exactly once
// with the final (result, err).
func runJob(
	ctx context.Context,
	svc ports.TranscribeService,
	req domain.Request,
	onProgress func(domain.ProgressEvent),
	onDone func(*domain.Result, error),
) (ports.Job, error) {
	job, err := svc.Submit(ctx, req)
	if err != nil {
		return nil, err
	}
	go func() {
		for ev := range job.Progress() {
			ev := ev // capture
			fyne.Do(func() { onProgress(ev) })
		}
		res, err := job.Wait()
		fyne.Do(func() { onDone(res, err) })
	}()
	return job, nil
}
