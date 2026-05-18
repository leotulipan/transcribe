package logging

import (
	"io"
	"log/slog"
	"os"

	"github.com/leotulipan/transcribe/internal/ports"
)

type slogLogger struct {
	inner *slog.Logger
}

func NewText(out io.Writer, level slog.Level) ports.Logger {
	if out == nil {
		out = os.Stderr
	}
	h := slog.NewTextHandler(out, &slog.HandlerOptions{Level: level})
	return &slogLogger{inner: slog.New(h)}
}

func NewJSON(out io.Writer, level slog.Level) ports.Logger {
	if out == nil {
		out = os.Stderr
	}
	h := slog.NewJSONHandler(out, &slog.HandlerOptions{Level: level})
	return &slogLogger{inner: slog.New(h)}
}

func NewDiscard() ports.Logger {
	return &slogLogger{inner: slog.New(slog.NewTextHandler(io.Discard, nil))}
}

func (l *slogLogger) Debug(msg string, kv ...any) { l.inner.Debug(msg, kv...) }
func (l *slogLogger) Info(msg string, kv ...any)  { l.inner.Info(msg, kv...) }
func (l *slogLogger) Warn(msg string, kv ...any)  { l.inner.Warn(msg, kv...) }
func (l *slogLogger) Error(msg string, kv ...any) { l.inner.Error(msg, kv...) }
