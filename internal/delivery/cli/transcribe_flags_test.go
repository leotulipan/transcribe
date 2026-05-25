package cli

import (
	"testing"

	"github.com/stretchr/testify/require"
)


func TestTranscribeCmd_HasPaddingStartFlag(t *testing.T) {
	cmd := newTranscribeCmd(Deps{})
	f := cmd.Flags().Lookup("padding-start")
	require.NotNil(t, f, "--padding-start flag must be registered")
	require.Equal(t, "0", f.DefValue, "default must be 0")
}

func TestTranscribeCmd_HasNewPhase5aFlags(t *testing.T) {
	cmd := newTranscribeCmd(Deps{})
	flags := []struct {
		name     string
		defValue string
	}{
		{"words-per-subtitle", "0"},
		{"silent-portions", "1500"},
		{"padding-end", "0"},
		{"show-pauses", "true"},
		{"start-hour", "0"},
	}
	for _, tc := range flags {
		t.Run(tc.name, func(t *testing.T) {
			f := cmd.Flags().Lookup(tc.name)
			require.NotNil(t, f, "--%s flag must be registered", tc.name)
			require.Equal(t, tc.defValue, f.DefValue, "--%s default mismatch", tc.name)
		})
	}
}

func TestTranscribeCmd_SilentPortionMsBackwardCompat(t *testing.T) {
	cmd := newTranscribeCmd(Deps{})
	f := cmd.Flags().Lookup("silent-portion-ms")
	require.NotNil(t, f, "--silent-portion-ms (legacy flag) must still be registered")
	require.Equal(t, "1500", f.DefValue)
}

func TestTranscribeCmd_WordsPerSubtitleAndCharsPerLineMutex(t *testing.T) {
	cmd := newTranscribeCmd(Deps{})
	// Both flags non-zero → error.
	require.NoError(t, cmd.Flags().Set("words-per-subtitle", "3"))
	require.NoError(t, cmd.Flags().Set("chars-per-line", "50"))
	err := cmd.RunE(cmd, []string{})
	require.Error(t, err)
	require.Contains(t, err.Error(), "mutually exclusive")
}

func TestTranscribeCmd_HasPhase5cFlags(t *testing.T) {
	cmd := newTranscribeCmd(Deps{})
	flags := []struct {
		name     string
		defValue string
	}{
		{"num-speakers", "0"},
		{"keyterms-prompt", ""},
		{"speech-models", ""},
	}
	for _, tc := range flags {
		t.Run(tc.name, func(t *testing.T) {
			f := cmd.Flags().Lookup(tc.name)
			require.NotNil(t, f, "--%s flag must be registered", tc.name)
			require.Equal(t, tc.defValue, f.DefValue, "--%s default mismatch", tc.name)
		})
	}
}

func TestTranscribeCmd_RejectsNumSpeakersOver32(t *testing.T) {
	cmd := newTranscribeCmd(Deps{})
	require.NoError(t, cmd.Flags().Set("num-speakers", "33"))
	err := cmd.RunE(cmd, []string{})
	require.Error(t, err)
	require.Contains(t, err.Error(), "num-speakers")
}

func TestTranscribeCmd_RejectsNegativeNumSpeakers(t *testing.T) {
	cmd := newTranscribeCmd(Deps{})
	require.NoError(t, cmd.Flags().Set("num-speakers", "-1"))
	err := cmd.RunE(cmd, []string{})
	require.Error(t, err)
	require.Contains(t, err.Error(), "num-speakers")
}

func TestTranscribeCmd_ParsesKeyTermsCommaList(t *testing.T) {
	result := parseCommaSeparated("foo, bar ,baz")
	require.Equal(t, []string{"foo", "bar", "baz"}, result)
}

func TestTranscribeCmd_ParseCommaSeparatedSkipsEmpties(t *testing.T) {
	result := parseCommaSeparated("foo,,  ,bar")
	require.Equal(t, []string{"foo", "bar"}, result)
}

func TestTranscribeCmd_ParseCommaSeparatedEmptyString(t *testing.T) {
	require.Nil(t, parseCommaSeparated(""))
}
