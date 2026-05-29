//go:build windows

// This file exists solely to host the //go:generate directive that produces
// resource.syso (icon + version metadata) for the GUI binary. The .syso file
// is linked automatically by `go build` when GOOS=windows.
//
// Regenerate with: go generate ./cmd/transcribe-gui
// (Normally invoked by scripts/build-installer.ps1.)

package main

//go:generate goversioninfo -64 -platform-specific=false
