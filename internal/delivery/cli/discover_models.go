package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"text/tabwriter"
	"time"

	"github.com/spf13/cobra"

	"github.com/leotulipan/transcribe/internal/adapters/api/elevenlabs"
	"github.com/leotulipan/transcribe/internal/adapters/api/gemini"
	"github.com/leotulipan/transcribe/internal/adapters/api/groq"
	"github.com/leotulipan/transcribe/internal/adapters/api/mistral"
	"github.com/leotulipan/transcribe/internal/adapters/api/openai"
	configadapter "github.com/leotulipan/transcribe/internal/adapters/config"
	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

// discoveryResult is one row of the discover-models output.
type discoveryResult struct {
	Provider  string   `json:"provider"`
	Count     int      `json:"count"`
	Models    []string `json:"models,omitempty"`
	ElapsedMS int64    `json:"elapsed_ms"`
	Status    string   `json:"status"`          // ok | missing | unsupported | error
	Error     string   `json:"error,omitempty"` // populated when Status != ok
}

type discoveryEnvelope struct {
	SavedTo string            `json:"saved_to,omitempty"`
	Results []discoveryResult `json:"results"`
}

func newDiscoverModelsCmd(d Deps) *cobra.Command {
	var (
		providerFilter string
		jsonOut        bool
		dryRun         bool
	)
	cmd := &cobra.Command{
		Use:   "discover-models",
		Short: "Fetch live model lists from each provider and cache them in config",
		Long: "Calls each provider's 'list models' endpoint, deduplicates and sorts the\n" +
			"result, and persists the lists under [discovered_models] in the user's\n" +
			"config.toml. Providers without a discovery endpoint (assemblyai) are\n" +
			"reported as 'unsupported' and skipped.\n\n" +
			"After running this, `transcribe transcribe --model X` and the GUI's model\n" +
			"dropdown will both show the live list instead of the bundled hardcoded one.",
		RunE: func(c *cobra.Command, _ []string) error {
			env := runDiscovery(c.Context(), d.Config, providerFilter)

			if !dryRun {
				// Merge new lists into a fresh load (so we don't clobber other
				// recent changes), then persist.
				store := configadapter.New()
				cfg, err := store.Load()
				if err != nil {
					return err
				}
				if cfg.DiscoveredModels == nil {
					cfg.DiscoveredModels = map[domain.ProviderID][]string{}
				}
				for _, r := range env.Results {
					if r.Status == "ok" && len(r.Models) > 0 {
						cfg.DiscoveredModels[domain.ProviderID(r.Provider)] = r.Models
					}
				}
				if err := store.Save(cfg); err != nil {
					return err
				}
				env.SavedTo = store.Path()
			}

			if jsonOut {
				enc := json.NewEncoder(os.Stdout)
				enc.SetIndent("", "  ")
				return enc.Encode(env)
			}
			renderDiscoveryTable(os.Stdout, env)
			for _, r := range env.Results {
				if r.Status == "error" {
					return errSilent
				}
			}
			return nil
		},
	}
	cmd.Flags().StringVar(&providerFilter, "provider", "",
		"only discover for this provider id (default: all)")
	cmd.Flags().BoolVar(&jsonOut, "json", false, "machine-readable output")
	cmd.Flags().BoolVar(&dryRun, "dry-run", false,
		"print results without writing to config.toml")
	return cmd
}

func runDiscovery(ctx context.Context, cfg ports.Config, only string) discoveryEnvelope {
	httpClient := &http.Client{Timeout: 30 * time.Second}
	out := make([]discoveryResult, 0, len(knownProviders))
	for _, id := range knownProviders {
		if only != "" && string(id) != only {
			continue
		}
		r := discoveryResult{Provider: string(id)}
		key := cfg.APIKeys[id]
		if key == "" {
			r.Status = "missing"
			r.Error = "no API key configured"
			out = append(out, r)
			continue
		}

		discoverer, ok := buildDiscoverer(id, key, httpClient)
		if !ok {
			r.Status = "unsupported"
			r.Error = "provider has no live model-list endpoint"
			out = append(out, r)
			continue
		}

		start := time.Now()
		list, err := discoverer.DiscoverModels(ctx)
		r.ElapsedMS = time.Since(start).Milliseconds()
		if err != nil {
			r.Status = "error"
			r.Error = truncate(err.Error(), 200)
		} else {
			r.Status = "ok"
			r.Models = list
			r.Count = len(list)
		}
		out = append(out, r)
	}
	return discoveryEnvelope{Results: out}
}

// buildDiscoverer constructs a per-provider client and returns it as a
// ModelDiscoverer if the adapter supports the interface. AssemblyAI has no
// live endpoint and returns (nil, false).
func buildDiscoverer(id domain.ProviderID, key string, h *http.Client) (ports.ModelDiscoverer, bool) {
	switch id {
	case domain.ProviderGroq:
		return groq.New(key, h), true
	case domain.ProviderOpenAI:
		return openai.New(key, h), true
	case domain.ProviderMistral:
		return mistral.New(key, h), true
	case domain.ProviderGemini:
		return gemini.New(key, h), true
	case domain.ProviderElevenLabs:
		return elevenlabs.New(key, h), true
	}
	return nil, false
}

func renderDiscoveryTable(w *os.File, env discoveryEnvelope) {
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "PROVIDER\tSTATUS\tCOUNT\tTIME\tNOTE")
	for _, r := range env.Results {
		note := r.Error
		if r.Status == "ok" {
			if len(r.Models) > 4 {
				note = fmt.Sprintf("%s, ...", joinFirst(r.Models, 4))
			} else {
				note = joinFirst(r.Models, len(r.Models))
			}
		}
		timeStr := "-"
		if r.ElapsedMS > 0 {
			timeStr = fmt.Sprintf("%d ms", r.ElapsedMS)
		}
		fmt.Fprintf(tw, "%s\t%s\t%d\t%s\t%s\n",
			r.Provider, statusLabel(r.Status), r.Count, timeStr, note)
	}
	_ = tw.Flush()
	if env.SavedTo != "" {
		fmt.Fprintf(w, "\nsaved to %s\n", env.SavedTo)
	}
}

func joinFirst(s []string, n int) string {
	if n > len(s) {
		n = len(s)
	}
	out := ""
	for i, v := range s[:n] {
		if i > 0 {
			out += ", "
		}
		out += v
	}
	return out
}

