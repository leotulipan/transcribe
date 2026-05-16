//go:build integration

package integration_test

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/stretchr/testify/require"
)

// Build the binary once and exercise it.
func TestCLI_JSONMode_HelpExits0(t *testing.T) {
	bin := buildBinary(t)
	out, err := exec.Command(bin, "--help").CombinedOutput()
	require.NoError(t, err, string(out))
	require.Contains(t, string(out), "transcribe")
}

func TestCLI_JSON_RejectsMissingApiKey(t *testing.T) {
	if os.Getenv("GROQ_API_KEY") != "" {
		t.Skip("real GROQ_API_KEY present — this test expects missing key")
	}
	bin := buildBinary(t)
	sample := mustTestdata(t, "short-sample.mp3")
	cmd := exec.Command(bin, "transcribe", "--json", "--api", "groq", "--output", "text", sample)
	cmd.Env = append(os.Environ(), "GROQ_API_KEY=")
	// Capture stdout and stderr separately so that log output on stderr
	// does not corrupt the JSON we expect on stdout.
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	_ = cmd.Run()
	out := stdout.Bytes()
	var got map[string]any
	require.NoError(t, json.Unmarshal(out, &got), "stdout: %s\nstderr: %s", string(out), stderr.String())
	require.Equal(t, "error", got["status"])
}

func buildBinary(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	name := "transcribe"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	bin := filepath.Join(dir, name)
	cmd := exec.Command("go", "build", "-o", bin, "../../cmd/transcribe")
	out, err := cmd.CombinedOutput()
	require.NoError(t, err, string(out))
	return bin
}

func mustTestdata(t *testing.T, name string) string {
	t.Helper()
	wd, err := os.Getwd()
	require.NoError(t, err)
	return filepath.Join(wd, "..", "..", "testdata", name)
}
