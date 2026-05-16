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

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	err = gui.Run(ctx, gui.Deps{Service: svc, Config: cfg, Logger: log, SaveConfig: saveCfg})
	if err != nil && !errors.Is(err, context.Canceled) {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(cli.ExitCodeFor(err))
	}
	_ = version
}
