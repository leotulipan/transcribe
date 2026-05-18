package cli

import (
	"encoding/json"
	"errors"
	"io"
	"time"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

const jsonSchemaVersion = 1

type jobLike interface {
	Progress() <-chan domain.ProgressEvent
	Wait() (*domain.Result, error)
}

type resultJSON struct {
	Provider   domain.ProviderID `json:"provider"`
	Model      string            `json:"model"`
	Language   string            `json:"language"`
	Text       string            `json:"text"`
	DurationMs int64             `json:"duration_ms"`
}

type errorJSON struct {
	Code    string         `json:"code"`
	Message string         `json:"message"`
	Details map[string]any `json:"details,omitempty"`
}

func renderJSON(w io.Writer, job jobLike, stream bool) error {
	enc := json.NewEncoder(w)
	if stream {
		for ev := range job.Progress() {
			_ = enc.Encode(map[string]any{
				"type":       "progress",
				"stage":      ev.Stage.String(),
				"percent":    ev.Percent,
				"elapsed_ms": ev.Elapsed.Milliseconds(),
				"message":    ev.Message,
			})
		}
		res, err := job.Wait()
		if err != nil {
			_ = enc.Encode(map[string]any{
				"type":  "error",
				"error": errorPayload(err),
			})
			return err
		}
		_ = enc.Encode(map[string]any{
			"type":   "result",
			"result": toResultJSON(res),
		})
		return nil
	}

	// Drain progress without emitting (final-only mode)
	for range job.Progress() {
	}
	res, err := job.Wait()
	if err != nil {
		_ = enc.Encode(map[string]any{
			"schema_version": jsonSchemaVersion,
			"status":         "error",
			"error":          errorPayload(err),
		})
		return err
	}
	_ = enc.Encode(map[string]any{
		"schema_version": jsonSchemaVersion,
		"status":         "ok",
		"result":         toResultJSON(res),
	})
	return nil
}

func toResultJSON(r *domain.Result) resultJSON {
	return resultJSON{
		Provider:   r.Provider,
		Model:      r.Model,
		Language:   r.Language,
		Text:       r.Text,
		DurationMs: int64(r.Duration / time.Millisecond),
	}
}

func errorPayload(err error) errorJSON {
	if err == nil {
		return errorJSON{}
	}
	var ei domain.ErrIncompatible
	if errors.As(err, &ei) {
		return errorJSON{
			Code: "incompatible", Message: err.Error(),
			Details: map[string]any{
				"provider": ei.Provider, "model": ei.Model, "format": ei.Format, "reason": ei.Reason,
			},
		}
	}
	var ep *domain.ErrProvider
	if errors.As(err, &ep) {
		return errorJSON{
			Code: "provider", Message: err.Error(),
			Details: map[string]any{"provider": ep.Provider, "status": ep.StatusCode, "retryable": ep.Retryable},
		}
	}
	if errors.Is(err, domain.ErrFFmpegMissing) {
		return errorJSON{Code: "ffmpeg_missing", Message: err.Error()}
	}
	if errors.Is(err, domain.ErrProviderMissing) {
		return errorJSON{Code: "provider_missing", Message: err.Error()}
	}
	return errorJSON{Code: "internal", Message: err.Error()}
}
