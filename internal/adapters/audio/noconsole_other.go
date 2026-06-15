//go:build !windows

package audio

import "os/exec"

// hideConsole is a no-op on non-Windows platforms, where child processes do
// not spawn console windows.
func hideConsole(cmd *exec.Cmd) {}
