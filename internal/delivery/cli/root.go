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
	var debug, verbose, showVersion bool
	root := &cobra.Command{
		Use:          "transcribe",
		Short:        "Transcribe audio and video files via multiple AI providers",
		SilenceUsage:  true,
		SilenceErrors: true,
		// Apply log-level flags before any subcommand runs; also handle -V/--version.
		PersistentPreRunE: func(cmd *cobra.Command, _ []string) error {
			if showVersion {
				if d.Version != "" {
					cmd.Println(d.Version)
				} else {
					cmd.Println("(dev)")
				}
				// Signal callers to exit cleanly.
				return nil
			}
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
	root.PersistentFlags().BoolVarP(&debug, "debug", "d", false, "enable debug-level logging")
	root.PersistentFlags().BoolVarP(&verbose, "verbose", "v", false, "enable info-level logging (default is warn)")
	// -V for --version frees -v for --verbose (cobra would auto-claim -v for version).
	root.Flags().BoolVarP(&showVersion, "version", "V", false, "show version and exit")
	root.AddCommand(newTranscribeCmd(d))
	root.AddCommand(newProvidersCmd(d))
	root.AddCommand(newSetupCmd(d))
	root.AddCommand(newTestKeysCmd(d))
	root.AddCommand(newDiscoverModelsCmd(d))
	return root
}
