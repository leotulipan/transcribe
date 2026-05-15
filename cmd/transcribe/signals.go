package main

import "context"

// installSignals is fleshed out in L5. Stub keeps the build green.
func installSignals(parent context.Context) (context.Context, context.CancelFunc) {
	return context.WithCancel(parent)
}
