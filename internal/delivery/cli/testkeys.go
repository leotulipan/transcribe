package cli

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/spf13/cobra"

	"github.com/leotulipan/transcribe/internal/adapters/api/assemblyai"
	"github.com/leotulipan/transcribe/internal/adapters/api/elevenlabs"
	"github.com/leotulipan/transcribe/internal/adapters/api/gemini"
	"github.com/leotulipan/transcribe/internal/adapters/api/groq"
	"github.com/leotulipan/transcribe/internal/adapters/api/mistral"
	"github.com/leotulipan/transcribe/internal/adapters/api/openai"
	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

// knownProviders lists every provider the binary knows about, in display order.
var knownProviders = []domain.ProviderID{
	domain.ProviderGroq,
	domain.ProviderOpenAI,
	domain.ProviderAssemblyAI,
	domain.ProviderElevenLabs,
	domain.ProviderGemini,
	domain.ProviderMistral,
}

type keyResult struct {
	Provider  string `json:"provider"`
	Status    string `json:"status"` // ok | invalid | missing | unsupported | error
	MaskedKey string `json:"masked_key,omitempty"`
	ElapsedMS int64  `json:"elapsed_ms"`
	Detail    string `json:"detail,omitempty"`
}

func newTestKeysCmd(d Deps) *cobra.Command {
	var jsonOut bool
	cmd := &cobra.Command{
		Use:   "test-keys",
		Short: "Validate configured API keys with non-consuming probes",
		Long: "Pings each provider with a free endpoint (typically GET /models) to verify " +
			"the configured key works. No transcription credit is consumed.",
		RunE: func(c *cobra.Command, _ []string) error {
			results := runKeyChecks(c.Context(), d.Config)
			if jsonOut {
				enc := json.NewEncoder(os.Stdout)
				enc.SetIndent("", "  ")
				return enc.Encode(results)
			}
			renderTable(os.Stdout, results)
			for _, r := range results {
				if r.Status == "invalid" || r.Status == "error" {
					return errSilent
				}
			}
			return nil
		},
	}
	cmd.Flags().BoolVar(&jsonOut, "json", false, "machine-readable output")
	return cmd
}

// errSilent triggers a non-zero exit via ExitCodeFor without re-printing a message
// (the table already shows the failure).
var errSilent = errors.New("one or more API keys failed validation")

func runKeyChecks(ctx context.Context, cfg ports.Config) []keyResult {
	httpClient := &http.Client{Timeout: 15 * time.Second}
	out := make([]keyResult, 0, len(knownProviders))
	for _, id := range knownProviders {
		key := cfg.APIKeys[id]
		r := keyResult{Provider: string(id), MaskedKey: maskKey(key)}
		if key == "" {
			r.Status = "missing"
			r.Detail = "no API key configured"
			out = append(out, r)
			continue
		}

		p := buildProvider(id, key, httpClient)
		if p == nil {
			r.Status = "error"
			r.Detail = "unknown provider"
			out = append(out, r)
			continue
		}

		start := time.Now()
		err := p.CheckKey(ctx)
		r.ElapsedMS = time.Since(start).Milliseconds()
		switch {
		case err == nil:
			r.Status = "ok"
		case errors.Is(err, errors.ErrUnsupported):
			r.Status = "unsupported"
			r.Detail = "CheckKey not implemented for this adapter"
		default:
			r.Status = "error"
			var pe *domain.ErrProvider
			if errors.As(err, &pe) && (pe.StatusCode == 401 || pe.StatusCode == 403) {
				r.Status = "invalid"
			}
			r.Detail = truncate(err.Error(), 200)
		}
		out = append(out, r)
	}
	return out
}

func buildProvider(id domain.ProviderID, key string, h *http.Client) ports.Provider {
	switch id {
	case domain.ProviderGroq:
		return groq.New(key, h)
	case domain.ProviderOpenAI:
		return openai.New(key, h)
	case domain.ProviderAssemblyAI:
		return assemblyai.New(key, h)
	case domain.ProviderElevenLabs:
		return elevenlabs.New(key, h)
	case domain.ProviderGemini:
		return gemini.New(key, h)
	case domain.ProviderMistral:
		return mistral.New(key, h)
	}
	return nil
}

func maskKey(k string) string {
	if k == "" {
		return ""
	}
	if len(k) < 8 {
		return "***"
	}
	return k[:3] + "..." + k[len(k)-4:]
}

func truncate(s string, n int) string {
	// Collapse any whitespace (newlines, tabs) so multi-line API error bodies
	// don't break the tabwriter table layout.
	s = strings.Join(strings.Fields(s), " ")
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

func renderTable(w *os.File, rows []keyResult) {
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "PROVIDER\tSTATUS\tKEY\tTIME\tDETAIL")
	for _, r := range rows {
		key := r.MaskedKey
		if key == "" {
			key = "-"
		}
		timeStr := "-"
		if r.ElapsedMS > 0 {
			timeStr = fmt.Sprintf("%d ms", r.ElapsedMS)
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\n",
			r.Provider, statusLabel(r.Status), key, timeStr, r.Detail)
	}
	_ = tw.Flush()
}

func statusLabel(s string) string {
	switch s {
	case "ok":
		return "OK"
	case "invalid":
		return "INVALID"
	case "missing":
		return "MISSING"
	case "unsupported":
		return "UNSUPPORTED"
	case "error":
		return "ERROR"
	}
	return s
}
