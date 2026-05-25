package cli

import (
	"bytes"
	"context"
	"log/slog"
	"testing"

	"github.com/spf13/cobra"
	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

// rootStubService satisfies ports.TranscribeService for root-level tests.
type rootStubService struct{}

func (rootStubService) ListProviders() []domain.ProviderID                              { return nil }
func (rootStubService) ListModels(_ domain.ProviderID) ([]string, error)                { return nil, nil }
func (rootStubService) DiscoverModels(_ context.Context, _ domain.ProviderID) ([]string, error) {
	return nil, nil
}
func (rootStubService) Submit(_ context.Context, _ domain.Request) (ports.Job, error) { return nil, nil }

// executeRootWithProbe builds a root command, attaches a no-op probe subcommand that
// fires PersistentPreRunE, then executes with the given args. Using a dedicated
// subcommand avoids the --help cobra short-circuit that bypasses PersistentPreRunE.
func executeRootWithProbe(t *testing.T, lv *slog.LevelVar, flags []string) error {
	t.Helper()
	d := Deps{Service: rootStubService{}, LevelVar: lv}
	root := NewRoot(d)
	// A minimal subcommand that simply returns nil — PersistentPreRunE fires before it.
	probe := &cobra.Command{
		Use:  "probe",
		RunE: func(_ *cobra.Command, _ []string) error { return nil },
	}
	root.AddCommand(probe)
	root.SetOut(&bytes.Buffer{})
	root.SetErr(&bytes.Buffer{})
	args := append(flags, "probe")
	root.SetArgs(args)
	return root.Execute()
}

func TestLogger_DefaultLevelIsWarn(t *testing.T) {
	lv := &slog.LevelVar{}
	lv.Set(slog.LevelWarn)
	err := executeRootWithProbe(t, lv, nil)
	require.NoError(t, err)
	require.Equal(t, slog.LevelWarn, lv.Level())
}

func TestLogger_VerboseFlagSetsInfoLevel(t *testing.T) {
	lv := &slog.LevelVar{}
	lv.Set(slog.LevelWarn)
	err := executeRootWithProbe(t, lv, []string{"--verbose"})
	require.NoError(t, err)
	require.Equal(t, slog.LevelInfo, lv.Level())
}

func TestLogger_DebugFlagSetsDebugLevel(t *testing.T) {
	lv := &slog.LevelVar{}
	lv.Set(slog.LevelWarn)
	err := executeRootWithProbe(t, lv, []string{"--debug"})
	require.NoError(t, err)
	require.Equal(t, slog.LevelDebug, lv.Level())
}

func TestLogger_DebugWinsOverVerbose(t *testing.T) {
	lv := &slog.LevelVar{}
	lv.Set(slog.LevelWarn)
	err := executeRootWithProbe(t, lv, []string{"--debug", "--verbose"})
	require.NoError(t, err)
	require.Equal(t, slog.LevelDebug, lv.Level())
}

func TestRoot_PersistentFlagsRegistered(t *testing.T) {
	root := NewRoot(Deps{})
	require.NotNil(t, root.PersistentFlags().Lookup("debug"), "--debug must be a persistent flag")
	require.NotNil(t, root.PersistentFlags().Lookup("verbose"), "--verbose must be a persistent flag")
}

func TestRoot_VersionFlagPrintsVersionAndExits(t *testing.T) {
	var buf bytes.Buffer
	root := NewRoot(Deps{Version: "test-1.2.3", LevelVar: new(slog.LevelVar)})
	root.SetOut(&buf)
	root.SetArgs([]string{"--version"})
	err := root.Execute()
	require.NoError(t, err)
	require.Contains(t, buf.String(), "test-1.2.3")
}

func TestRoot_VersionFlagShortAliasPrintsVersionAndExits(t *testing.T) {
	var buf bytes.Buffer
	root := NewRoot(Deps{Version: "test-2.0.0", LevelVar: new(slog.LevelVar)})
	root.SetOut(&buf)
	root.SetArgs([]string{"-V"})
	err := root.Execute()
	require.NoError(t, err)
	require.Contains(t, buf.String(), "test-2.0.0")
}

func TestRoot_NoArgsShowsHelp(t *testing.T) {
	var buf bytes.Buffer
	root := NewRoot(Deps{Version: "1.0.0", LevelVar: new(slog.LevelVar)})
	root.SetOut(&buf)
	root.SetArgs([]string{})
	err := root.Execute()
	require.NoError(t, err)
	require.Contains(t, buf.String(), "Transcribe audio and video files via multiple AI providers")
}
