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
