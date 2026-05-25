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
	api              string
	model            string
	language         string
	outputs          []string
	outDir           string
	cache            bool
	davinci          bool
	wordSRT          bool
	silentMs         int
	silentPortions   int  // alias for silentMs; last one set wins
	paddingStartMs   int
	paddingEndMs     int
	jsonMode         bool
	progress         bool
	fillerWords      []string
	removeFillers    bool
	fillerLines      bool
	charsPerLine     int
	wordsPerSubtitle int
	showPauses       bool
	startHour        int
	diarize          bool
	speakerLabels    bool
	fps              float64
	fpsOffsetStart   int
	fpsOffsetEnd     int
	numSpeakers      int
	keyTermsPrompt   string
	speechModels     string

	// Audio pipeline knobs (Phase 5d).
	sizeThresholdMB float64 // stored as MB; converted to bytes when building Request
	chunkLengthSec  int
	overlapSec      int
	useInput        bool
	usePCM          bool
	keep            bool
	keepFLAC        bool

	// I/O & workflow flags (Phase 5e).
	force           bool   // force re-transcription even when a sidecar exists (sets UseCache=false)
	saveCleanedJSON bool   // persist normalized JSON even when UseCache=false
	useJSONInput    bool   // treat input path as a pre-saved sidecar JSON; skip API call
	extensions      string // comma-separated extension filter for directory enumeration

	// Logging & discovery flags (Phase 5f).
	list bool // print providers+models and exit without transcribing
}

func newTranscribeCmd(d Deps) *cobra.Command {
	f := &transcribeFlags{}
	cmd := &cobra.Command{
		Use:   "transcribe [flags] <file> [<file>...]",
		Short: "Transcribe one or more files",
		Args:  cobra.ArbitraryArgs,
		RunE: func(c *cobra.Command, args []string) error {
			// --list: print providers and exit without touching any inputs.
			if f.list {
				printProviders(d, os.Stdout)
				return nil
			}
			// Mirror semantic: --speaker-labels defaults to --diarize when not explicitly set.
			if !c.Flags().Changed("speaker-labels") {
				f.speakerLabels = f.diarize
			}
			// --silent-portions is an alias for --silent-portion-ms; last one set wins.
			if c.Flags().Changed("silent-portions") {
				f.silentMs = f.silentPortions
			}
			// Mutex: --words-per-subtitle and --chars-per-line cannot be combined.
			if f.wordsPerSubtitle > 0 && f.charsPerLine > 0 {
				return fmt.Errorf("--words-per-subtitle and --chars-per-line are mutually exclusive")
			}
			// Validate num-speakers range before any TUI escalation so the error
			// surfaces unconditionally (even when no input files are provided).
			if f.numSpeakers < 0 || f.numSpeakers > 32 {
				return fmt.Errorf("--num-speakers must be between 0 and 32 (got %d)", f.numSpeakers)
			}
			// Mutex: --use-input and --use-pcm are contradictory.
			if f.useInput && f.usePCM {
				return fmt.Errorf("--use-input and --use-pcm are mutually exclusive")
			}
			if f.chunkLengthSec < 0 {
				return fmt.Errorf("--chunk-length must be 0 (unset) or positive (got %d)", f.chunkLengthSec)
			}
			if f.overlapSec < 0 {
				return fmt.Errorf("--overlap must be 0 (no overlap) or positive (got %d)", f.overlapSec)
			}
			if f.sizeThresholdMB < 0 {
				return fmt.Errorf("--size-threshold must be 0 (unset) or positive (got %g)", f.sizeThresholdMB)
			}
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
	cmd.Flags().StringVarP(&f.api, "api", "a", string(d.Config.DefaultProvider), "transcription API id")
	cmd.Flags().StringVarP(&f.model, "model", "m", "", "model name (provider default if empty)")
	cmd.Flags().StringVarP(&f.language, "language", "l", d.Config.DefaultLanguage, "ISO-639-1 language hint")
	cmd.Flags().StringSliceVarP(&f.outputs, "output", "o", []string{"text"}, "output formats: text,srt,word_srt,davinci_srt")
	cmd.Flags().StringVar(&f.outDir, "output-dir", "", "output directory (default: next to input)")
	cmd.Flags().BoolVar(&f.cache, "use-cache", true, "reuse sidecar transcripts when present")
	cmd.Flags().BoolVarP(&f.davinci, "davinci", "D", false, "convenience flag: enable davinci_srt output")
	cmd.Flags().BoolVarP(&f.wordSRT, "word-srt", "C", false, "convenience flag: enable word_srt output")
	cmd.Flags().IntVar(&f.silentMs, "silent-portion-ms", 1500, "pause threshold for davinci mode (ms)")
	cmd.Flags().IntVarP(&f.silentPortions, "silent-portions", "p", 1500, "pause threshold for davinci mode (ms); alias for --silent-portion-ms")
	cmd.Flags().IntVar(&f.paddingStartMs, "padding-start", 0, "shift subtitle starts earlier by up to this many ms (davinci mode)")
	cmd.Flags().IntVar(&f.paddingEndMs, "padding-end", 0, "shrink subtitle end times earlier by up to this many ms (davinci mode)")
	cmd.Flags().IntVarP(&f.wordsPerSubtitle, "words-per-subtitle", "w", 0, "max words per subtitle block (0 = default 7); mutually exclusive with --chars-per-line")
	cmd.Flags().BoolVar(&f.showPauses, "show-pauses", true, "emit (...) markers for pauses >= --silent-portions (davinci mode)")
	cmd.Flags().IntVar(&f.startHour, "start-hour", 0, "hour offset added to all SRT/DaVinci timecodes")
	cmd.Flags().BoolVar(&f.jsonMode, "json", false, "agent-callable JSON output, no TUI escalation")
	cmd.Flags().BoolVar(&f.progress, "progress", false, "with --json, emit JSONL progress events")
	cmd.Flags().StringSliceVar(&f.fillerWords, "filler-words", nil, "comma-separated filler words (default: um,uh,ähm,äh,hm,hmm)")
	cmd.Flags().BoolVar(&f.removeFillers, "remove-fillers", false, "drop filler words from output entirely")
	cmd.Flags().BoolVar(&f.fillerLines, "filler-lines", true, "uppercase fillers so DaVinci renders them on their own line")
	cmd.Flags().IntVarP(&f.charsPerLine, "chars-per-line", "c", 0, "max chars per rendered subtitle line (0 = no wrapping)")
	cmd.Flags().BoolVar(&f.diarize, "diarize", false, "request speaker diarization from the provider (assemblyai, elevenlabs)")
	cmd.Flags().BoolVar(&f.speakerLabels, "speaker-labels", false, "prefix subtitle blocks with [Speaker X]: (default: mirrors --diarize)")
	cmd.Flags().Float64Var(&f.fps, "fps", 0, "video frame rate for snapping subtitle boundaries to frame grid (0 = no snapping)")
	cmd.Flags().IntVar(&f.fpsOffsetStart, "fps-offset-start", -1, "frame offset added to snapped Start times (-1 = appear 1 frame earlier)")
	cmd.Flags().IntVar(&f.fpsOffsetEnd, "fps-offset-end", 0, "frame offset added to snapped End times")
	cmd.Flags().IntVar(&f.numSpeakers, "num-speakers", 0, "expected number of speakers 1..32 (assemblyai+elevenlabs; requires --diarize; 0 = unset)")
	cmd.Flags().StringVar(&f.keyTermsPrompt, "keyterms-prompt", "", "comma-separated key terms to boost recognition (assemblyai)")
	cmd.Flags().StringVar(&f.speechModels, "speech-models", "", "comma-separated fallback speech models (assemblyai)")
	// Audio pipeline flags (Phase 5d).
	cmd.Flags().Float64Var(&f.sizeThresholdMB, "size-threshold", 100, "files (MB) under this size skip conversion if format is compatible (0 = provider limit only)")
	cmd.Flags().IntVar(&f.chunkLengthSec, "chunk-length", 0, "chunk duration in seconds (0 = derive from byte budget)")
	cmd.Flags().IntVar(&f.overlapSec, "overlap", 0, "overlap in seconds between consecutive chunks (0 = no overlap)")
	cmd.Flags().BoolVar(&f.useInput, "use-input", false, "bypass conversion — send original file as-is")
	cmd.Flags().BoolVar(&f.usePCM, "use-pcm", false, "convert to PCM WAV instead of the preferred codec")
	cmd.Flags().BoolVar(&f.keep, "keep", false, "retain all intermediate files instead of deleting them")
	cmd.Flags().BoolVar(&f.keepFLAC, "keep-flac", false, "retain FLAC intermediate files instead of deleting them")
	// I/O & workflow flags (Phase 5e).
	cmd.Flags().BoolVarP(&f.force, "force", "r", false, "re-transcribe even when a sidecar transcript exists (overrides --use-cache)")
	cmd.Flags().BoolVarP(&f.saveCleanedJSON, "save-cleaned-json", "J", false, "persist the normalized pre-format JSON next to outputs even when --use-cache=false")
	cmd.Flags().BoolVarP(&f.useJSONInput, "use-json-input", "j", false, "accept a previously-saved sidecar JSON as input and skip the API call")
	cmd.Flags().StringVarP(&f.extensions, "extensions", "e", "", "comma-separated file extensions to filter directory enumeration (e.g. mp3,m4a)")
	// Logging & discovery flags (Phase 5f).
	cmd.Flags().BoolVar(&f.list, "list", false, "list available APIs and their models, then exit")
	return cmd
}

func runTranscribe(ctx context.Context, d Deps, f *transcribeFlags, files []string) error {
	formats, err := parseFormats(f.outputs, f.davinci, f.wordSRT)
	if err != nil {
		return err
	}
	keyTerms := parseCommaSeparated(f.keyTermsPrompt)
	speechModels := parseCommaSeparated(f.speechModels)

	// --use-json-input + --force is contradictory: there is no cache lookup to bypass.
	if f.useJSONInput && f.force {
		return fmt.Errorf("--use-json-input and --force are mutually exclusive: no API call is made with --use-json-input")
	}

	// Validate: --use-json-input paths must end in .json.
	if f.useJSONInput {
		for _, p := range files {
			if !strings.HasSuffix(strings.ToLower(p), ".json") {
				return fmt.Errorf("--use-json-input: path must be a .json sidecar file, got: %s", p)
			}
		}
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

	var expanded []string
	if f.useJSONInput {
		// JSON-input paths are passed verbatim — no audio enumeration needed.
		expanded = files
	} else {
		// Expand any directory arguments into their constituent audio/video files.
		// Files are passed through unchanged. This lets users do
		//     transcribe path/to/folder file.mp3 anotherFolder/
		exts := parseExtensions(f.extensions)
		expanded, err = expandPathsWith(files, exts)
		if err != nil {
			return err
		}
		if len(expanded) == 0 {
			return fmt.Errorf("no audio/video files found in: %s", strings.Join(files, ", "))
		}
	}

	useCache := f.cache
	if f.force {
		useCache = false
	}

	provider := domain.ProviderID(f.api)
	for _, file := range expanded {
		req := domain.Request{
			InputPath:        file,
			Provider:         provider,
			Model:            f.model,
			Language:         f.language,
			Formats:          formats,
			OutputDir:        f.outDir,
			UseCache:         useCache,
			MaxCharsPerLine:  f.charsPerLine,
			WordsPerSubtitle: f.wordsPerSubtitle,
			StartHour:        f.startHour,
			SpeakerLabels:    f.speakerLabels, // f.speakerLabels mirrors f.diarize unless --speaker-labels is explicit
			NumSpeakers:      f.numSpeakers,
			KeyTerms:         keyTerms,
			SpeechModels:     speechModels,
			// Audio pipeline (Phase 5d).
			SizeThresholdBytes:    int64(f.sizeThresholdMB * 1024 * 1024),
			ChunkLengthSec:        f.chunkLengthSec,
			OverlapSec:            f.overlapSec,
			UseInput:              f.useInput,
			UsePCM:                f.usePCM,
			KeepIntermediates:     f.keep,
			KeepFLACIntermediates: f.keepFLAC,
			// I/O & workflow (Phase 5e).
			UseJSONInput:    f.useJSONInput,
			SaveCleanedJSON: f.saveCleanedJSON,
		}
		if hasFormat(formats, domain.FormatDavinciSRT) {
			opts := &domain.DaVinciOptions{
				SilentPortionThreshold: parseSilentMs(f.silentMs),
				PaddingStart:           parseSilentMs(f.paddingStartMs),
				PaddingEnd:             parseSilentMs(f.paddingEndMs),
				RemoveFillers:          f.removeFillers,
				SuppressFillerLines:    !f.fillerLines,
				SuppressPauses:         !f.showPauses,
				FPS:                    f.fps,
				FPSOffsetStart:         f.fpsOffsetStart,
				FPSOffsetEnd:           f.fpsOffsetEnd,
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

func parseFormats(outs []string, davinciFlag bool, wordSRTFlag bool) ([]domain.OutputFormat, error) {
	seen := map[domain.OutputFormat]bool{}
	var out []domain.OutputFormat
	for _, name := range outs {
		for _, raw := range strings.Split(name, ",") {
			f := domain.OutputFormat(strings.TrimSpace(strings.ToLower(raw)))
			switch f {
			case domain.FormatText, domain.FormatSRT, domain.FormatWordSRT, domain.FormatDavinciSRT:
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
	if wordSRTFlag && !seen[domain.FormatWordSRT] {
		out = append(out, domain.FormatWordSRT)
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

// expandPathsWith walks each input using the provided extension filter.
// Files pass through; directories are expanded via EnumerateAudioFilesWith.
// Pass nil extensions to use the default AudioExtensions list.
func expandPathsWith(paths []string, extensions []string) ([]string, error) {
	var out []string
	for _, p := range paths {
		more, err := services.EnumerateAudioFilesWith(p, extensions)
		if err != nil {
			return nil, fmt.Errorf("enumerate %s: %w", p, err)
		}
		out = append(out, more...)
	}
	return out, nil
}

// parseExtensions splits a comma-separated extension string into a normalised
// slice (lower-cased, leading dot optional). Returns nil when s is empty.
func parseExtensions(s string) []string {
	if s == "" {
		return nil
	}
	raw := strings.Split(s, ",")
	out := make([]string, 0, len(raw))
	for _, e := range raw {
		e = strings.TrimSpace(strings.ToLower(e))
		if e == "" {
			continue
		}
		if !strings.HasPrefix(e, ".") {
			e = "." + e
		}
		out = append(out, e)
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func firstOr(s []string, def string) string {
	if len(s) > 0 {
		return s[0]
	}
	return def
}

// mustFormats parses formats from flags, returning an empty slice on error.
func mustFormats(f *transcribeFlags) []domain.OutputFormat {
	out, _ := parseFormats(f.outputs, f.davinci, f.wordSRT)
	return out
}

// parseCommaSeparated splits a comma-delimited string into trimmed, non-empty entries.
func parseCommaSeparated(s string) []string {
	if s == "" {
		return nil
	}
	raw := strings.Split(s, ",")
	out := make([]string, 0, len(raw))
	for _, part := range raw {
		if t := strings.TrimSpace(part); t != "" {
			out = append(out, t)
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}
