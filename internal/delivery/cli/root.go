package cli

import (
	"log/slog"

	"github.com/spf13/cobra"

	"github.com/leotulipan/transcribe/internal/ports"
)

type Deps struct {
	Service  ports.TranscribeService
	Config   ports.Config
	Logger   ports.Logger
	Version  string
	LevelVar *slog.LevelVar
}

func NewRoot(d Deps) *cobra.Command {
	var debug, verbose bool
	root := &cobra.Command{
		Use:           "transcribe",
		Short:         "Transcribe audio and video files via multiple AI providers",
		Version:       d.Version,
		SilenceUsage:  true,
		SilenceErrors: true,
		// Apply log-level flags before any subcommand runs.
		PersistentPreRunE: func(_ *cobra.Command, _ []string) error {
			if d.LevelVar == nil {
				return nil
			}
			switch {
			case debug:
				d.LevelVar.Set(slog.LevelDebug)
			case verbose:
				d.LevelVar.Set(slog.LevelInfo)
			default:
				d.LevelVar.Set(slog.LevelWarn)
			}
			return nil
		},
	}
	root.PersistentFlags().BoolVar(&debug, "debug", false, "enable debug-level logging")
	root.PersistentFlags().BoolVar(&verbose, "verbose", false, "enable info-level logging (default is warn)")
	root.AddCommand(newTranscribeCmd(d))
	root.AddCommand(newProvidersCmd(d))
	root.AddCommand(newSetupCmd(d))
	root.AddCommand(newTestKeysCmd(d))
	root.AddCommand(newDiscoverModelsCmd(d))
	return root
}
