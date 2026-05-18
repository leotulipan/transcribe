package delivery

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/adapters/api/assemblyai"
	"github.com/leotulipan/transcribe/internal/adapters/api/elevenlabs"
	"github.com/leotulipan/transcribe/internal/adapters/api/gemini"
	"github.com/leotulipan/transcribe/internal/adapters/api/groq"
	"github.com/leotulipan/transcribe/internal/adapters/api/mistral"
	"github.com/leotulipan/transcribe/internal/adapters/api/openai"
)

// Mirrors tests/unit/test_audio_processing.py:test_api_file_size_limits.
// Locks the per-provider upload caps so chunking decisions stay correct.
func TestProviderMaxUploadBytes(t *testing.T) {
	const (
		mb = 1024 * 1024
		gb = 1024 * mb
	)
	cases := []struct {
		name string
		got  int64
		want int64
	}{
		{"groq", groq.New("", nil).MaxUploadBytes(), 25 * mb},
		{"openai", openai.New("", nil).MaxUploadBytes(), 25 * mb},
		{"mistral", mistral.New("", nil).MaxUploadBytes(), 25 * mb},
		{"assemblyai", assemblyai.New("", nil).MaxUploadBytes(), 200 * mb},
		{"elevenlabs", elevenlabs.New("", nil).MaxUploadBytes(), 1000 * mb},
		{"gemini", gemini.New("", nil).MaxUploadBytes(), 2 * gb},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			require.Equal(t, tc.want, tc.got)
		})
	}
}
