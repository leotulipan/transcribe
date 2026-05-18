package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"runtime"

	"github.com/leotulipan/transcribe/internal/adapters/config"
	"github.com/leotulipan/transcribe/internal/adapters/logging"
	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/delivery"
	"github.com/leotulipan/transcribe/internal/delivery/cli"
	"github.com/leotulipan/transcribe/internal/delivery/gui"
	"github.com/leotulipan/transcribe/internal/delivery/tui"
	"github.com/leotulipan/transcribe/internal/ports"
)

var version = "dev"

type uiMode int

const (
	modeCLI uiMode = iota
	modeTUI
	modeGUI
	modeJSON
)

func decideMode(args []string) uiMode {
	hasJSON := false
	explicit := ""
	for _, a := range args[1:] {
		if a == "--json" {
			hasJSON = true
		}
		if a == "--ui=tui" {
			explicit = "tui"
		}
		if a == "--ui=gui" {
			explicit = "gui"
		}
		if a == "--ui=cli" {
			explicit = "cli"
		}
	}
	if hasJSON {
		return modeJSON
	}
	switch explicit {
	case "tui":
		return modeTUI
	case "gui":
		return modeGUI
	case "cli":
		return modeCLI
	}
	if len(args) == 1 {
		// Zero args: Linux headless → TUI; Windows → GUI.
		if runtime.GOOS == "linux" && os.Getenv("DISPLAY") == "" {
			return modeTUI
		}
		if runtime.GOOS == "windows" {
			return modeGUI
		}
		return modeTUI
	}
	return modeCLI
}

func main() {
	cfg, err := config.New().Load()
	if err != nil {
		fmt.Fprintln(os.Stderr, "config:", err)
		os.Exit(3)
	}
	log := logging.NewText(os.Stderr, slog.LevelInfo)

	svc, err := delivery.BuildService(cfg, log)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(3)
	}

	ctx, cancel := installSignals(context.Background())
	defer cancel()

	mode := decideMode(os.Args)
	switch mode {
	case modeGUI:
		saveCfg := func(c ports.Config) error { return config.New().Save(c) }
		loadCfg := func() (ports.Config, error) { return config.New().Load() }
		buildSvc := func(c ports.Config) (ports.TranscribeService, error) {
			return delivery.BuildService(c, log)
		}
		deps := gui.NewDeps(svc, cfg, log, saveCfg, loadCfg, buildSvc)
		err = gui.Run(ctx, deps)
		if err != nil && !errors.Is(err, context.Canceled) {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(cli.ExitCodeFor(err))
		}
	case modeTUI:
		_, err = tui.Run(ctx, tui.Deps{Service: svc, Config: cfg, Logger: log}, tui.Prefill{})
		if err != nil && !errors.Is(err, context.Canceled) {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(cli.ExitCodeFor(err))
		}
	default:
		root := cli.NewRoot(cli.Deps{Service: svc, Config: cfg, Logger: log, Version: version})
		if err := root.ExecuteContext(ctx); err != nil {
			// CLI may signal escalation via a typed error
			var esc *cli.EscalateToTUI
			if errors.As(err, &esc) {
				_, e := tui.Run(ctx, tui.Deps{Service: svc, Config: cfg, Logger: log}, tui.Prefill{
					InputPath: esc.InputPath,
					Provider:  domain.ProviderID(esc.Provider),
					Model:     esc.Model,
					Language:  esc.Language,
					Formats:   esc.Formats,
				})
				if e != nil && !errors.Is(e, context.Canceled) {
					fmt.Fprintln(os.Stderr, e)
					os.Exit(cli.ExitCodeFor(e))
				}
				return
			}
			fmt.Fprintln(os.Stderr, err)
			os.Exit(cli.ExitCodeFor(err))
		}
	}
}
