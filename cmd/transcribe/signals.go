package main

import (
	"context"
	"os"
	"os/signal"
	"syscall"
)

// installSignals returns a context that cancels on SIGINT or SIGTERM.
func installSignals(parent context.Context) (context.Context, context.CancelFunc) {
	ctx, cancel := context.WithCancel(parent)
	ch := make(chan os.Signal, 1)
	signal.Notify(ch, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-ch
		cancel()
		signal.Stop(ch)
	}()
	return ctx, cancel
}
