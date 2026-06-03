package format

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

func TestText_Write(t *testing.T) {
	dir := t.TempDir()
	dst := filepath.Join(dir, "out.txt")
	w := NewText()
	require.Equal(t, domain.FormatText, w.Format())
	require.NoError(t, w.Write(&domain.Result{Text: "hello world\n"}, dst, domain.WriteOpts{}))
	got, err := os.ReadFile(dst)
	require.NoError(t, err)
	require.Equal(t, "hello world\n", string(got))
}

func TestText_SpeakerLabels(t *testing.T) {
	res := &domain.Result{
		Text: "Hello world Goodbye",
		Words: []domain.Word{
			{Text: "Hello", Start: 100 * time.Millisecond, Speaker: "0"},
			{Text: "world", Start: 600 * time.Millisecond, Speaker: "0"},
			{Text: "Goodbye", Start: 5000 * time.Millisecond, Speaker: "1"},
		},
	}

	t.Run("enabled_groups_by_turn", func(t *testing.T) {
		dir := t.TempDir()
		dst := filepath.Join(dir, "out.txt")
		require.NoError(t, NewText().Write(res, dst, domain.WriteOpts{SpeakerLabels: true}))
		got, _ := os.ReadFile(dst)
		s := string(got)
		require.Contains(t, s, "[Speaker 0]: Hello world")
		require.Contains(t, s, "[Speaker 1]: Goodbye")
	})

	t.Run("named_labels", func(t *testing.T) {
		named := &domain.Result{Words: []domain.Word{
			{Text: "Hallo", Speaker: "Julia"},
			{Text: "Servus", Start: 5 * time.Second, Speaker: "Gast"},
		}}
		dir := t.TempDir()
		dst := filepath.Join(dir, "out.txt")
		require.NoError(t, NewText().Write(named, dst, domain.WriteOpts{SpeakerLabels: true}))
		got, _ := os.ReadFile(dst)
		s := string(got)
		require.Contains(t, s, "[Julia]: Hallo")
		require.Contains(t, s, "[Gast]: Servus")
	})

	t.Run("disabled_uses_plain_text", func(t *testing.T) {
		dir := t.TempDir()
		dst := filepath.Join(dir, "out.txt")
		require.NoError(t, NewText().Write(res, dst, domain.WriteOpts{SpeakerLabels: false}))
		got, _ := os.ReadFile(dst)
		require.False(t, strings.Contains(string(got), "[Speaker"))
	})
}
