package cli

import (
	"log/slog"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestTranscribeCmd_ShortFlagAliases(t *testing.T) {
	cmd := newTranscribeCmd(Deps{})
	cases := []struct{ short, long string }{
		{"a", "api"},
		{"l", "language"},
		{"o", "output"},
		{"c", "chars-per-line"},
		{"C", "word-srt"},
		{"D", "davinci"},
		{"m", "model"},
		{"p", "silent-portions"},
		{"w", "words-per-subtitle"},
		{"e", "extensions"},
		{"r", "force"},
		{"j", "use-json-input"},
		{"J", "save-cleaned-json"},
	}
	for _, tc := range cases {
		t.Run(tc.long, func(t *testing.T) {
			long := cmd.Flags().Lookup(tc.long)
			require.NotNil(t, long, "long flag missing: %s", tc.long)
			require.Equal(t, tc.short, long.Shorthand, "short flag mismatch for --%s", tc.long)
		})
	}
}

func TestRoot_ShortFlagAliases(t *testing.T) {
	root := NewRoot(Deps{LevelVar: new(slog.LevelVar)})
	cases := []struct{ short, long string }{
		{"d", "debug"},
		{"v", "verbose"},
		{"V", "version"},
	}
	for _, tc := range cases {
		t.Run(tc.long, func(t *testing.T) {
			long := root.Flags().Lookup(tc.long)
			if long == nil {
				long = root.PersistentFlags().Lookup(tc.long)
			}
			require.NotNil(t, long, "flag missing: %s", tc.long)
			require.Equal(t, tc.short, long.Shorthand, "short flag mismatch for --%s", tc.long)
		})
	}
}
