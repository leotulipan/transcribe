package audio

import (
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

func TestMeta_RoundTrip(t *testing.T) {
	dir := t.TempDir()
	intermediate := filepath.Join(dir, "x.m4a")

	in := MetaInfo{
		SchemaVersion:   metaSchema,
		Operation:       "copy",
		SourcePath:      "C:/videos/in.mp4",
		SourceSize:      12345,
		SourceMTimeUnix: 1700000000,
		TargetCodec:     "aac",
		TargetContainer: "m4a",
		MaxBytesBudget:  25 * 1024 * 1024,
		Provider:        domain.ProviderGroq,
		Model:           "whisper-large-v3",
	}
	require.NoError(t, WriteMeta(intermediate, in))

	out, err := ReadMeta(intermediate)
	require.NoError(t, err)
	require.Equal(t, in, out)
}
