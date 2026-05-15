package cli

import "github.com/spf13/cobra"

func newSetupCmd(d Deps) *cobra.Command {
	return &cobra.Command{Use: "setup", Short: "non-interactive setup (impl in L6)"}
}
