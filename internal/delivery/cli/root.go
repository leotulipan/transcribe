package cli

import (
	"github.com/spf13/cobra"

	"github.com/leotulipan/transcribe/internal/ports"
)

type Deps struct {
	Service ports.TranscribeService
	Config  ports.Config
	Logger  ports.Logger
	Version string
}

func NewRoot(d Deps) *cobra.Command {
	root := &cobra.Command{
		Use:           "transcribe",
		Short:         "Transcribe audio and video files via multiple AI providers",
		Version:       d.Version,
		SilenceUsage:  true,
		SilenceErrors: true,
	}
	root.AddCommand(newTranscribeCmd(d))
	root.AddCommand(newProvidersCmd(d))
	root.AddCommand(newSetupCmd(d))
	return root
}
