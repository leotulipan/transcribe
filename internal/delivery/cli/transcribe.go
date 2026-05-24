package cli

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/core/services"
)

type transcribeFlags struct {
	api           string
	model         string
	language      string
	outputs       []string
	outDir        string
	cache         bool
	davinci       bool
	silentMs      int
	paddingStartMs int
	jsonMode      bool
	progress      bool
	fillerWords   []string
	removeFillers bool
	fillerLines   bool
}

func newTranscribeCmd(d Deps) *cobra.Command {
	f := &transcribeFlags{}
	cmd := &cobra.Command{
		Use:   "transcribe [flags] <file> [<file>...]",
		Short: "Transcribe one or more files",
		Args:  cobra.ArbitraryArgs,
		RunE: func(c *cobra.Command, args []string) error {
			if len(args) == 0 && f.jsonMode {
				return fmt.Errorf("at least one input file is required in --json mode")
			}
			if len(args) == 0 {
				return &EscalateToTUI{
					Provider: f.api,
					Model:    f.model,
					Language: f.language,
					Formats:  mustFormats(f),
				}
			}
			return runTranscribe(c.Context(), d, f, args)
		},
	}
	cmd.Flags().StringVar(&f.api, "api", string(d.Config.DefaultProvider), "transcription API id")
	cmd.Flags().StringVar(&f.model, "model", "", "model name (provider default if empty)")
	cmd.Flags().StringVar(&f.language, "language", d.Config.DefaultLanguage, "ISO-639-1 language hint")
	cmd.Flags().StringSliceVar(&f.outputs, "output", []string{"text"}, "output formats: text,srt,davinci_srt")
	cmd.Flags().StringVar(&f.outDir, "output-dir", "", "output directory (default: next to input)")
	cmd.Flags().BoolVar(&f.cache, "use-cache", true, "reuse sidecar transcripts when present")
	cmd.Flags().BoolVar(&f.davinci, "davinci", false, "convenience flag: enable davinci_srt output")
	cmd.Flags().IntVar(&f.silentMs, "silent-portion-ms", 1500, "pause threshold for davinci mode")
	cmd.Flags().IntVar(&f.paddingStartMs, "padding-start", 0, "shift subtitle starts earlier by up to this many ms (davinci mode, default 0)")
	cmd.Flags().BoolVar(&f.jsonMode, "json", false, "agent-callable JSON output, no TUI escalation")
	cmd.Flags().BoolVar(&f.progress, "progress", false, "with --json, emit JSONL progress events")
	cmd.Flags().StringSliceVar(&f.fillerWords, "filler-words", nil, "comma-separated filler words (default: um,uh,ähm,äh,hm,hmm)")
	cmd.Flags().BoolVar(&f.removeFillers, "remove-fillers", false, "drop filler words from output entirely")
	cmd.Flags().BoolVar(&f.fillerLines, "filler-lines", true, "uppercase fillers so DaVinci renders them on their own line")
	return cmd
}

func runTranscribe(ctx context.Context, d Deps, f *transcribeFlags, files []string) error {
	formats, err := parseFormats(f.outputs, f.davinci)
	if err != nil {
		return err
	}
	if !f.jsonMode {
		// Escalation rule: if no provider is configured at all, escalate to TUI.
		if f.api == "" && d.Config.DefaultProvider == "" {
			return &EscalateToTUI{
				InputPath: firstOr(files, ""),
				Model:     f.model,
				Language:  f.language,
				Formats:   formats,
			}
		}
	}
	// Expand any directory arguments into their constituent audio/video files.
	// Files are passed through unchanged. This lets users do
	//     transcribe path/to/folder file.mp3 anotherFolder/
	expanded, err := expandPaths(files)
	if err != nil {
		return err
	}
	if len(expanded) == 0 {
		return fmt.Errorf("no audio/video files found in: %s", strings.Join(files, ", "))
	}

	provider := domain.ProviderID(f.api)
	for _, file := range expanded {
		req := domain.Request{
			InputPath: file,
			Provider:  provider,
			Model:     f.model,
			Language:  f.language,
			Formats:   formats,
			OutputDir: f.outDir,
			UseCache:  f.cache,
		}
		if hasFormat(formats, domain.FormatDavinciSRT) {
			opts := &domain.DaVinciOptions{
				SilentPortionThreshold: parseSilentMs(f.silentMs),
				PaddingStart:           parseSilentMs(f.paddingStartMs),
				RemoveFillers:          f.removeFillers,
				SuppressFillerLines:    !f.fillerLines,
			}
			if len(f.fillerWords) > 0 {
				opts.FillerWords = f.fillerWords
			}
			req.DaVinciOpts = opts
		}
		if err := submitOne(ctx, d, req, f); err != nil {
			return err
		}
	}
	return nil
}

func parseFormats(outs []string, davinciFlag bool) ([]domain.OutputFormat, error) {
	seen := map[domain.OutputFormat]bool{}
	var out []domain.OutputFormat
	for _, name := range outs {
		for _, raw := range strings.Split(name, ",") {
			f := domain.OutputFormat(strings.TrimSpace(strings.ToLower(raw)))
			switch f {
			case domain.FormatText, domain.FormatSRT, domain.FormatDavinciSRT:
			default:
				return nil, fmt.Errorf("unknown output format %q", raw)
			}
			if !seen[f] {
				seen[f] = true
				out = append(out, f)
			}
		}
	}
	if davinciFlag && !seen[domain.FormatDavinciSRT] {
		out = append(out, domain.FormatDavinciSRT)
	}
	return out, nil
}

func hasFormat(fs []domain.OutputFormat, target domain.OutputFormat) bool {
	for _, f := range fs {
		if f == target {
			return true
		}
	}
	return false
}

func submitOne(ctx context.Context, d Deps, req domain.Request, f *transcribeFlags) error {
	job, err := d.Service.Submit(ctx, req)
	if err != nil {
		return err
	}
	if f.jsonMode {
		return renderJSON(os.Stdout, job, f.progress)
	}
	return renderText(os.Stderr, job)
}

func renderText(stderr *os.File, job interface {
	Progress() <-chan domain.ProgressEvent
	Wait() (*domain.Result, error)
}) error {
	for ev := range job.Progress() {
		fmt.Fprintf(stderr, "[%s] %s\n", ev.Stage, ev.Message)
	}
	_, err := job.Wait()
	return err
}

func parseSilentMs(ms int) time.Duration { return time.Duration(ms) * time.Millisecond }

// EscalateToTUI is returned by the CLI when required inputs are missing under
// non-JSON mode. main.go catches it and launches the TUI prefilled with what
// the user did pass.
type EscalateToTUI struct {
	InputPath string
	Provider  string
	Model     string
	Language  string
	Formats   []domain.OutputFormat
}

func (e *EscalateToTUI) Error() string { return "escalating to TUI for missing inputs" }

// expandPaths walks each input. Files pass through; directories are
// expanded via EnumerateAudioFiles. Order matches the input.
func expandPaths(paths []string) ([]string, error) {
	var out []string
	for _, p := range paths {
		more, err := services.EnumerateAudioFiles(p)
		if err != nil {
			return nil, fmt.Errorf("enumerate %s: %w", p, err)
		}
		out = append(out, more...)
	}
	return out, nil
}

func firstOr(s []string, def string) string {
	if len(s) > 0 {
		return s[0]
	}
	return def
}

// mustFormats parses formats from flags, returning an empty slice on error.
func mustFormats(f *transcribeFlags) []domain.OutputFormat {
	out, _ := parseFormats(f.outputs, f.davinci)
	return out
}
