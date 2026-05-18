package format

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

func TestText_Write(t *testing.T) {
	dir := t.TempDir()
	dst := filepath.Join(dir, "out.txt")
	w := NewText()
	require.Equal(t, domain.FormatText, w.Format())
	require.NoError(t, w.Write(&domain.Result{Text: "hello world\n"}, dst))
	got, err := os.ReadFile(dst)
	require.NoError(t, err)
	require.Equal(t, "hello world\n", string(got))
}
