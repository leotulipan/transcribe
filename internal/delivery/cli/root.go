package cli

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/spf13/cobra"

	"github.com/leotulipan/transcribe/internal/adapters/config"
	"github.com/leotulipan/transcribe/internal/delivery/tui"
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
	var debug, verbose, showVersion, setupMode bool
	root := &cobra.Command{
		Use:          "transcribe",
		Short:        "Transcribe audio and video files via multiple AI providers",
		SilenceUsage:  true,
		SilenceErrors: true,
		// Handle version and setup flags at root level (no subcommand).
		RunE: func(cmd *cobra.Command, args []string) error {
			if showVersion {
				if d.Version != "" {
					fmt.Fprintln(cmd.OutOrStdout(), d.Version)
				} else {
					fmt.Fprintln(cmd.OutOrStdout(), "(dev)")
				}
				return nil
			}
			if setupMode {
				return runSetupWizard(d)
			}
			// No flags and no subcommand: show help.
			return cmd.Help()
		},
		// Apply log-level flags before any subcommand runs.
		PersistentPreRunE: func(cmd *cobra.Command, _ []string) error {
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
	root.PersistentFlags().BoolVar(&setupMode, "setup", false, "launch interactive API key setup wizard")
	root.AddCommand(newTranscribeCmd(d))
	root.AddCommand(newProvidersCmd(d))
	root.AddCommand(newSetupCmd(d))
	root.AddCommand(newTestKeysCmd(d))
	root.AddCommand(newDiscoverModelsCmd(d))
	return root
}

// runSetupWizard launches the interactive TUI wizard, then persists the result.
func runSetupWizard(d Deps) error {
	tuiDeps := tui.Deps{
		Service: d.Service,
		Config:  d.Config,
		Logger:  d.Logger,
	}
	newCfg, err := tui.RunWizard(context.Background(), tuiDeps)
	if err != nil {
		return err
	}
	store := config.New()
	if saveErr := store.Save(newCfg); saveErr != nil {
		return saveErr
	}
	fmt.Println("wrote", store.Path())
	return nil
}
