package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"

	"github.com/leotulipan/transcribe/internal/adapters/config"
	"github.com/leotulipan/transcribe/internal/adapters/logging"
	"github.com/leotulipan/transcribe/internal/delivery"
	"github.com/leotulipan/transcribe/internal/delivery/cli"
)

var version = "dev"

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

	root := cli.NewRoot(cli.Deps{Service: svc, Config: cfg, Logger: log, Version: version})
	ctx, cancel := installSignals(context.Background())
	defer cancel()
	if err := root.ExecuteContext(ctx); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(cli.ExitCodeFor(err))
	}
}
