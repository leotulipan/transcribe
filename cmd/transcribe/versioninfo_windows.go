//go:build windows

// This file exists solely to host the //go:generate directive that produces
// resource.syso (icon + version metadata) for the Windows build of the CLI.
// The .syso file is linked automatically by `go build` when GOOS=windows.
//
// Regenerate with: go generate ./cmd/transcribe
// (Normally invoked by scripts/build-installer.ps1.)

package main

//go:generate goversioninfo -64 -platform-specific=false
