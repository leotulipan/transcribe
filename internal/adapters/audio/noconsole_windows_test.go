//go:build windows

package audio

import (
	"os/exec"
	"testing"
)

func TestHideConsole_SetsCreateNoWindow(t *testing.T) {
	cmd := exec.Command("ffmpeg")
	hideConsole(cmd)
	if cmd.SysProcAttr == nil {
		t.Fatal("SysProcAttr was not set")
	}
	const createNoWindow = 0x08000000
	if cmd.SysProcAttr.CreationFlags&createNoWindow == 0 {
		t.Fatalf("CREATE_NO_WINDOW bit not set: got flags %#x", cmd.SysProcAttr.CreationFlags)
	}
}
