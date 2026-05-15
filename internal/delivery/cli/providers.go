package cli

import "github.com/spf13/cobra"

func newProvidersCmd(d Deps) *cobra.Command {
	return &cobra.Command{Use: "providers", Short: "list configured providers (impl in L6)"}
}
