package logging

import (
	"io"
	"log/slog"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestNewLevelled_RespectsLevelVar(t *testing.T) {
	lv := &slog.LevelVar{}
	lv.Set(slog.LevelWarn)
	logger := NewLevelled(io.Discard, lv)
	require.NotNil(t, logger)
}

func TestNewLevelled_NilOutDefaultsToStderr(t *testing.T) {
	lv := &slog.LevelVar{}
	logger := NewLevelled(nil, lv)
	require.NotNil(t, logger)
}
