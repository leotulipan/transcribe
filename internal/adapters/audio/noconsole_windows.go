//go:build windows

package audio

import (
	"os/exec"
	"syscall"
)

// hideConsole stops a child process (ffmpeg/ffprobe) from opening its own
// console window. When the GUI binary is built with -H windowsgui it has no
// console of its own, so each console child would otherwise flash a black
// window. CREATE_NO_WINDOW (0x08000000) runs the child without one.
func hideConsole(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{
		HideWindow:    true,
		CreationFlags: 0x08000000, // CREATE_NO_WINDOW
	}
}
