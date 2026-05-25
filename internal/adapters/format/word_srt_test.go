package format

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

// Word-level SRT output (one subtitle per word, useful for tight caption sync).
// Python source: audio_transcribe/utils/formatters.py — create_srt(format_type="word")

func TestWordSRT_OneSubtitlePerWord(t *testing.T) {
	res := &domain.Result{
		Words: []domain.Word{
			{Text: "Hello", Start: 1000 * time.Millisecond, End: 1500 * time.Millisecond},
			{Text: "world", Start: 1600 * time.Millisecond, End: 2100 * time.Millisecond},
		},
	}
	dir := t.TempDir()
	dst := filepath.Join(dir, "out.srt")
	require.NoError(t, NewWordSRT().Write(res, dst, domain.WriteOpts{}))

	got, err := os.ReadFile(dst)
	require.NoError(t, err)

	want := "1\n00:00:01,000 --> 00:00:01,500\nHello\n\n2\n00:00:01,600 --> 00:00:02,100\nworld\n\n"
	require.Equal(t, want, string(got))
}

func TestWordSRT_EmptyResultProducesEmptyFile(t *testing.T) {
	res := &domain.Result{}
	dir := t.TempDir()
	dst := filepath.Join(dir, "out.srt")
	require.NoError(t, NewWordSRT().Write(res, dst, domain.WriteOpts{}))

	got, err := os.ReadFile(dst)
	require.NoError(t, err)
	require.Empty(t, got)
}

func TestWordSRT_PreservesPunctuationOnWord(t *testing.T) {
	res := &domain.Result{
		Words: []domain.Word{
			{Text: "Hello,", Start: 1000 * time.Millisecond, End: 1500 * time.Millisecond},
		},
	}
	dir := t.TempDir()
	dst := filepath.Join(dir, "out.srt")
	require.NoError(t, NewWordSRT().Write(res, dst, domain.WriteOpts{}))

	got, err := os.ReadFile(dst)
	require.NoError(t, err)

	want := "1\n00:00:01,000 --> 00:00:01,500\nHello,\n\n"
	require.Equal(t, want, string(got))
}

func TestWordSRT_StartHourShiftsTimecodes(t *testing.T) {
	res := &domain.Result{Words: []domain.Word{
		{Text: "Hello", Start: 1 * time.Second, End: 2 * time.Second},
	}}
	dir := t.TempDir()
	dst := filepath.Join(dir, "out.srt")
	require.NoError(t, NewWordSRT().Write(res, dst, domain.WriteOpts{StartHour: 1}))
	body, err := os.ReadFile(dst)
	require.NoError(t, err)
	s := string(body)
	require.Contains(t, s, "01:00:01,000 --> 01:00:02,000")
}
