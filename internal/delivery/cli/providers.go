package cli

import (
	"encoding/json"
	"fmt"
	"io"
	"os"

	"github.com/spf13/cobra"
)

// printProviders writes the text-format provider+model list to w.
// Shared by the providers subcommand and the --list flag on transcribe.
func printProviders(d Deps, w io.Writer) {
	for _, id := range d.Service.ListProviders() {
		models, _ := d.Service.ListModels(id)
		fmt.Fprintln(w, string(id))
		for _, m := range models {
			fmt.Fprintln(w, "  -", m)
		}
	}
}

func newProvidersCmd(d Deps) *cobra.Command {
	var jsonOut bool
	cmd := &cobra.Command{
		Use:   "providers",
		Short: "List configured providers and their models",
		RunE: func(c *cobra.Command, _ []string) error {
			if jsonOut {
				type entry struct {
					Provider string   `json:"provider"`
					Models   []string `json:"models"`
				}
				var entries []entry
				for _, id := range d.Service.ListProviders() {
					models, _ := d.Service.ListModels(id)
					entries = append(entries, entry{Provider: string(id), Models: models})
				}
				enc := json.NewEncoder(os.Stdout)
				enc.SetIndent("", "  ")
				return enc.Encode(entries)
			}
			printProviders(d, os.Stdout)
			return nil
		},
	}
	cmd.Flags().BoolVar(&jsonOut, "json", false, "machine-readable output")
	return cmd
}
