package cli

import (
	"bytes"
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

type fakeJob struct {
	events []domain.ProgressEvent
	res    *domain.Result
	err    error
}

func (f *fakeJob) Progress() <-chan domain.ProgressEvent {
	ch := make(chan domain.ProgressEvent, len(f.events))
	for _, ev := range f.events {
		ch <- ev
	}
	close(ch)
	return ch
}
func (f *fakeJob) Wait() (*domain.Result, error) { return f.res, f.err }

func TestRenderJSON_FinalOnly_Success(t *testing.T) {
	job := &fakeJob{res: &domain.Result{Text: "hi", Provider: domain.ProviderGroq}}
	var buf bytes.Buffer
	require.NoError(t, renderJSON(&buf, job, false))
	var got map[string]any
	require.NoError(t, json.Unmarshal(buf.Bytes(), &got))
	require.Equal(t, float64(1), got["schema_version"])
	require.Equal(t, "ok", got["status"])
	require.NotNil(t, got["result"])
}

func TestRenderJSON_FinalOnly_Error(t *testing.T) {
	job := &fakeJob{err: domain.ErrIncompatible{Provider: domain.ProviderGroq, Format: domain.FormatSRT, Reason: "no timestamps"}}
	var buf bytes.Buffer
	require.Error(t, renderJSON(&buf, job, false))
	var got map[string]any
	require.NoError(t, json.Unmarshal(buf.Bytes(), &got))
	require.Equal(t, "error", got["status"])
}

func TestRenderJSON_Stream_EmitsProgressLines(t *testing.T) {
	job := &fakeJob{
		events: []domain.ProgressEvent{
			{Stage: domain.StageProbing, Elapsed: 10 * time.Millisecond},
			{Stage: domain.StageTranscribing, Percent: 0.5, Elapsed: 50 * time.Millisecond},
		},
		res: &domain.Result{Text: "ok"},
	}
	var buf bytes.Buffer
	require.NoError(t, renderJSON(&buf, job, true))
	lines := bytes.Split(bytes.TrimRight(buf.Bytes(), "\n"), []byte("\n"))
	require.GreaterOrEqual(t, len(lines), 3) // 2 progress + 1 result
	var last map[string]any
	require.NoError(t, json.Unmarshal(lines[len(lines)-1], &last))
	require.Equal(t, "result", last["type"])
}

func TestRenderJSON_Stream_EmitsErrorLine(t *testing.T) {
	boom := errors.New("nope")
	job := &fakeJob{err: boom}
	var buf bytes.Buffer
	require.Error(t, renderJSON(&buf, job, true))
	require.Contains(t, buf.String(), `"type":"error"`)
}
