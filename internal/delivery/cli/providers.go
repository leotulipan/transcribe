package cli

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

func newProvidersCmd(d Deps) *cobra.Command {
	var jsonOut bool
	cmd := &cobra.Command{
		Use:   "providers",
		Short: "List configured providers and their models",
		RunE: func(c *cobra.Command, _ []string) error {
			type entry struct {
				Provider string   `json:"provider"`
				Models   []string `json:"models"`
			}
			var entries []entry
			for _, id := range d.Service.ListProviders() {
				models, _ := d.Service.ListModels(id)
				entries = append(entries, entry{Provider: string(id), Models: models})
			}
			if jsonOut {
				enc := json.NewEncoder(os.Stdout)
				enc.SetIndent("", "  ")
				return enc.Encode(entries)
			}
			for _, e := range entries {
				fmt.Println(e.Provider)
				for _, m := range e.Models {
					fmt.Println("  -", m)
				}
			}
			return nil
		},
	}
	cmd.Flags().BoolVar(&jsonOut, "json", false, "machine-readable output")
	return cmd
}
