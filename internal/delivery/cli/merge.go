package cli

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/leotulipan/transcribe/internal/adapters/format"
	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/core/services"
)

type mergeFlags struct {
	api          string
	model        string
	language     string
	outDir       string
	cache        bool
	charsPerLine int
	speakers     []string // repeated LABEL=PATH
	offsets      []string // repeated LABEL=DURATION
}

func newMergeCmd(d Deps) *cobra.Command {
	f := &mergeFlags{}
	cmd := &cobra.Command{
		Use:   "merge --speaker LABEL=FILE --speaker LABEL=FILE [flags]",
		Short: "Transcribe separate per-speaker tracks and merge into one labeled transcript",
		Long: "Transcribe two or more single-speaker recordings (e.g. one mic per podcast\n" +
			"participant) separately, writing per-track SRT/JSON/text, then interleave them\n" +
			"by timestamp into a combined SRT + text transcript labeled with each speaker's name.",
		Args: cobra.NoArgs,
		RunE: func(c *cobra.Command, _ []string) error {
			return runMerge(c.Context(), d, f)
		},
	}
	cmd.Flags().StringVarP(&f.api, "api", "a", string(d.Config.DefaultProvider), "transcription API id")
	cmd.Flags().StringVarP(&f.model, "model", "m", "", "model name (provider default if empty)")
	cmd.Flags().StringVarP(&f.language, "language", "l", d.Config.DefaultLanguage, "ISO-639-1 language hint")
	cmd.Flags().StringVar(&f.outDir, "output-dir", "", "output directory (default: next to inputs)")
	cmd.Flags().BoolVar(&f.cache, "use-cache", true, "reuse sidecar transcripts when present")
	cmd.Flags().IntVarP(&f.charsPerLine, "chars-per-line", "c", 0, "max chars per rendered subtitle line (0 = no wrapping)")
	cmd.Flags().StringArrayVar(&f.speakers, "speaker", nil, "speaker track as LABEL=FILE (repeat for each track; 2+ required)")
	cmd.Flags().StringArrayVar(&f.offsets, "offset", nil, "per-track time offset as LABEL=DURATION (e.g. Gast=1.2s); optional")
	return cmd
}

func runMerge(ctx context.Context, d Deps, f *mergeFlags) error {
	if len(f.speakers) < 2 {
		return fmt.Errorf("merge requires at least two --speaker LABEL=FILE tracks")
	}
	tracks, err := parseSpeakerTracks(f.speakers)
	if err != nil {
		return err
	}
	offsets, err := parseOffsets(f.offsets)
	if err != nil {
		return err
	}
	for label := range offsets {
		if !hasLabel(tracks, label) {
			return fmt.Errorf("--offset references unknown speaker label %q", label)
		}
	}

	useCache := f.cache
	provider := domain.ProviderID(f.api)

	labeled := make([]services.LabeledTrack, 0, len(tracks))
	for _, tr := range tracks {
		req := domain.Request{
			InputPath:       tr.path,
			Provider:        provider,
			Model:           f.model,
			Language:        f.language,
			Formats:         []domain.OutputFormat{domain.FormatSRT, domain.FormatText},
			OutputDir:       f.outDir,
			UseCache:        useCache,
			MaxCharsPerLine: f.charsPerLine,
			SaveCleanedJSON: true, // per-track JSON sidecar
		}
		job, jerr := d.Service.Submit(ctx, req)
		if jerr != nil {
			return jerr
		}
		for ev := range job.Progress() {
			fmt.Fprintf(os.Stderr, "[%s] %s %s\n", tr.label, ev.Stage, ev.Message)
		}
		res, werr := job.Wait()
		if werr != nil {
			return fmt.Errorf("transcribe %s (%s): %w", tr.label, tr.path, werr)
		}
		labeled = append(labeled, services.LabeledTrack{
			Result: res,
			Label:  tr.label,
			Offset: offsets[tr.label],
		})
	}

	merged := services.MergeResults(labeled)

	base := combinedBasePath(tracks, f.outDir)
	writeOpts := domain.WriteOpts{MaxCharsPerLine: f.charsPerLine, SpeakerLabels: true}
	if err := format.NewSRT().Write(merged, base+".srt", writeOpts); err != nil {
		return fmt.Errorf("write combined srt: %w", err)
	}
	if err := format.NewText().Write(merged, base+".txt", writeOpts); err != nil {
		return fmt.Errorf("write combined text: %w", err)
	}
	fmt.Fprintf(os.Stderr, "wrote combined %s.srt and %s.txt\n", base, base)
	return nil
}

type speakerTrack struct {
	label string
	path  string
}

func parseSpeakerTracks(pairs []string) ([]speakerTrack, error) {
	out := make([]speakerTrack, 0, len(pairs))
	seen := map[string]bool{}
	for _, p := range pairs {
		label, path, ok := splitLabelValue(p)
		if !ok {
			return nil, fmt.Errorf("--speaker must be LABEL=FILE, got %q", p)
		}
		if seen[label] {
			return nil, fmt.Errorf("--speaker label %q used more than once", label)
		}
		if _, err := os.Stat(path); err != nil {
			return nil, fmt.Errorf("--speaker %q: %w", label, err)
		}
		seen[label] = true
		out = append(out, speakerTrack{label: label, path: path})
	}
	return out, nil
}

func parseOffsets(pairs []string) (map[string]time.Duration, error) {
	out := map[string]time.Duration{}
	for _, p := range pairs {
		label, val, ok := splitLabelValue(p)
		if !ok {
			return nil, fmt.Errorf("--offset must be LABEL=DURATION, got %q", p)
		}
		dur, err := time.ParseDuration(val)
		if err != nil {
			return nil, fmt.Errorf("--offset %q: invalid duration %q: %w", label, val, err)
		}
		out[label] = dur
	}
	return out, nil
}

// splitLabelValue splits "LABEL=VALUE" on the first '='. Trims whitespace around
// the label; the value is kept verbatim (paths may contain '=' only after the
// first, which we preserve).
func splitLabelValue(s string) (label, value string, ok bool) {
	i := strings.IndexByte(s, '=')
	if i <= 0 {
		return "", "", false
	}
	label = strings.TrimSpace(s[:i])
	value = s[i+1:]
	if label == "" || value == "" {
		return "", "", false
	}
	return label, value, true
}

func hasLabel(tracks []speakerTrack, label string) bool {
	for _, t := range tracks {
		if t.label == label {
			return true
		}
	}
	return false
}

// combinedBasePath derives the base path (without extension) for the combined
// output files from the common prefix of the track basenames. Falls back to
// "<firstBase>_combined" when there is no shared prefix.
func combinedBasePath(tracks []speakerTrack, outDir string) string {
	dir := outDir
	if dir == "" {
		dir = filepath.Dir(tracks[0].path)
	}
	bases := make([]string, len(tracks))
	for i, t := range tracks {
		b := filepath.Base(t.path)
		bases[i] = strings.TrimSuffix(b, filepath.Ext(b))
	}
	prefix := strings.TrimRight(commonPrefix(bases), "_-. ")
	if prefix == "" {
		prefix = bases[0]
	}
	return filepath.Join(dir, prefix+"_combined")
}

func commonPrefix(s []string) string {
	if len(s) == 0 {
		return ""
	}
	p := s[0]
	for _, x := range s[1:] {
		max := len(p)
		if len(x) < max {
			max = len(x)
		}
		i := 0
		for i < max && p[i] == x[i] {
			i++
		}
		p = p[:i]
	}
	return p
}
