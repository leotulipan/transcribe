//go:build windows

package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"

	"github.com/leotulipan/transcribe/internal/adapters/config"
	"github.com/leotulipan/transcribe/internal/adapters/logging"
	"github.com/leotulipan/transcribe/internal/delivery"
	"github.com/leotulipan/transcribe/internal/delivery/cli"
	"github.com/leotulipan/transcribe/internal/delivery/gui"
	"github.com/leotulipan/transcribe/internal/ports"
)

var version = "dev"

func main() {
	cfg, err := config.New().Load()
	if err != nil {
		os.Exit(3)
	}
	log := logging.NewText(os.Stderr, slog.LevelInfo)
	svc, err := delivery.BuildService(cfg, log)
	if err != nil {
		os.Exit(3)
	}
	saveCfg := func(c ports.Config) error { return config.New().Save(c) }
	loadCfg := func() (ports.Config, error) { return config.New().Load() }
	buildSvc := func(c ports.Config) (ports.TranscribeService, error) {
		return delivery.BuildService(c, log)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	deps := gui.NewDeps(svc, cfg, log, saveCfg, loadCfg, buildSvc, version)
	// A file handed to us by Windows (drag onto the shortcut, or right-click
	// "Transcribe with…") arrives as argv; pre-fill the picker with it.
	deps.InitialPath = gui.FirstFileArg(os.Args[1:])
	err = gui.Run(ctx, deps)
	if err != nil && !errors.Is(err, context.Canceled) {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(cli.ExitCodeFor(err))
	}
}
