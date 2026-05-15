# Go Port of Transcribe — Hexagonal Design Spec

**Date:** 2026-05-15
**Author:** Leonard Tulipan + Claude (brainstorming session)
**Status:** Draft, awaiting user review

## 1. Goal

Replace the existing Python `transcribe` tool with a Go reimplementation that ships as a single binary and exposes three delivery surfaces — CLI, TUI, GUI — backed by a single hexagonal core. The Go port lives in a git worktree off this repo; once Go reaches the parity bar defined in section 2, a follow-up PR removes the Python tree entirely from `main`.

## 2. Scope

### v1 (this spec)

- **Providers:** AssemblyAI, ElevenLabs, Groq, OpenAI, Gemini, Mistral — all six.
- **Output formats:** plain text, standard SRT, DaVinci-Resolve SRT (with pause `(...)` markers and uppercase filler-word lines).
- **Audio handling:** ffmpeg via `os/exec`, automatic chunking for size-limited providers (Groq/OpenAI 25 MB), automatic audio extraction from video.
- **Caching:** sidecar JSON next to input; cache hits skip the API call.
- **Capability enforcement:** model-level capabilities checked before any work, hard error when the requested format needs timestamps and the model can't produce them.
- **UIs:** Cobra CLI (with agent-friendly `--json` mode), Bubble Tea TUI, Fyne GUI. Single executable; `--ui=<x>` switches mode, with auto-escalation rules defined in section 7.
- **Config:** TOML at `%LOCALAPPDATA%\transcribe\config.toml` (Windows v1 location).
- **Build target:** Windows amd64 only. Linux/macOS scaffolding present (build tags, CI structure) but not released.

### v2 / deferred

- macOS + Linux builds, macOS signing/notarization.
- Speaker diarization (ElevenLabs).
- Word-level SRT output (`word_srt`).
- Drag-and-drop batch templates.
- Interactive `--setup` wizard polish (v1 ships a non-interactive `transcribe setup` subcommand for `--json` users; the TUI's settings screen covers the interactive path).
- ElevenLabs speaker labels in SRT (`--speaker-labels`).
- DaVinci SRT advanced timing knobs beyond `--silent-portion`.

## 3. Architecture

Strict hexagonal layout. Three layers; dependencies point inward only.

```
.                              # repo root, single go.mod
├── go.mod / go.sum
├── cmd/
│   ├── transcribe/            # main entry point — one binary
│   │   └── main.go            # parses --ui, dispatches to cli/tui/gui
│   └── transcribe-gui/        # Windows -H windowsgui wrapper (no console)
│       └── main.go
├── internal/
│   ├── core/
│   │   ├── domain/            # plain structs + sentinel errors, no deps
│   │   └── services/          # transcription orchestrator + pipeline
│   ├── ports/                 # interfaces only (input + output ports)
│   ├── adapters/
│   │   ├── api/               # one subpkg per provider
│   │   │   ├── assemblyai/  elevenlabs/  groq/  openai/  gemini/  mistral/
│   │   │   └── internal/retry/  # shared HTTP retry helper
│   │   ├── audio/             # ffmpeg / ffprobe wrapper
│   │   ├── config/            # TOML store
│   │   ├── cache/             # sidecar JSON cache
│   │   ├── format/            # text / srt / davinci writers
│   │   └── logging/           # log/slog handler
│   └── delivery/              # composition root + UIs
│       ├── wire.go            # BuildService — only place adapters get wired
│       ├── cli/               # Cobra commands + JSON renderer
│       ├── tui/               # Bubble Tea models
│       └── gui/               # Fyne windows
├── docs/superpowers/specs/    # this spec
├── docs/                      # user-facing docs (migrated from Python repo)
└── testdata/                  # fixtures: small audio + golden outputs
```

**Layer rules:**

- `core/domain` imports only stdlib.
- `core/services` imports `core/domain` and `ports`. Never an adapter, never `fmt.Print`.
- `ports` imports only `core/domain`.
- `adapters/*` imports `ports` + `core/domain` + its own external deps. Never another adapter, never `core/services`, never `delivery`.
- `delivery/*` is the only place that imports concrete adapters. UIs see only `ports.TranscribeService`.

## 4. Domain types (`internal/core/domain`)

```go
package domain

import "time"

type ProviderID string
type OutputFormat string

const (
    FormatText       OutputFormat = "text"
    FormatSRT        OutputFormat = "srt"
    FormatDavinciSRT OutputFormat = "davinci_srt"
)

const (
    ProviderAssemblyAI ProviderID = "assemblyai"
    ProviderElevenLabs ProviderID = "elevenlabs"
    ProviderGroq       ProviderID = "groq"
    ProviderOpenAI     ProviderID = "openai"
    ProviderGemini     ProviderID = "gemini"
    ProviderMistral    ProviderID = "mistral"
)

type Request struct {
    InputPath   string
    Provider    ProviderID
    Model       string             // "" → provider default
    Language    string             // ISO-639-1; "" → auto-detect
    Formats     []OutputFormat
    OutputDir   string             // "" → next to input
    DaVinciOpts *DaVinciOptions    // nil unless DavinciSRT requested
    UseCache    bool
}

type Result struct {
    Provider   ProviderID
    Model      string
    Language   string
    Text       string
    Confidence float64
    Words      []Word
    Segments   []Segment
    Speakers   []Speaker          // empty in v1
    Duration   time.Duration
    SourcePath string
    RawJSON    []byte             // pristine provider response
}

type Word    struct { Text string; Start, End time.Duration; Confidence float64 }
type Segment struct { Text string; Start, End time.Duration; SpeakerID string }
type Speaker struct { ID, Label string }

type DaVinciOptions struct {
    SilentPortionThreshold time.Duration
    PaddingStart           time.Duration
    FillerWords            []string  // default: um, uh, ähm, äh, hm, hmm
}

type AudioFile struct {
    Path      string
    SizeBytes int64
    Duration  time.Duration
    Codec     string
    IsTemp    bool          // managed-temp file; cleanup deletes
    Chunks    []Chunk       // populated only when chunking applied
}

type Chunk struct {
    Path        string
    StartOffset time.Duration
    SizeBytes   int64
}

type ProgressEvent struct {
    Stage   Stage
    Message string
    Percent float64           // -1 when not estimable
    Elapsed time.Duration
}

type Stage int
const (
    StageProbing Stage = iota
    StageExtracting
    StageCompressing
    StageChunking
    StageUploading
    StageTranscribing
    StageParsing
    StageWriting
    StageDone
)
```

**Departures from the Python `TranscriptionResult`:**

1. Timestamps are `time.Duration`, not float seconds — kills a class of unit bugs.
2. `Result.RawJSON` carries the unmodified provider response. The cache writes this verbatim so we never lose information to a "cleaned" intermediate.

### 4.1 Sentinel errors (`domain/errors.go`)

```go
var (
    ErrConfigMissing   = errors.New("config error")
    ErrProviderMissing = errors.New("provider not configured")
    ErrFFmpegMissing   = errors.New("ffmpeg not found")
    ErrCanceled        = errors.New("canceled")
)

type ErrIncompatible struct {
    Provider ProviderID
    Model    string
    Format   OutputFormat
    Reason   string
}

type ErrProvider struct {
    Provider   ProviderID
    StatusCode int       // 0 if not HTTP
    Retryable  bool
    Cause      error
}
```

Errors are wrapped with context via `fmt.Errorf("%w: ...", base)` as they bubble up. A single `cli.exitCode(err)` helper maps these to documented exit codes.

## 5. Ports (`internal/ports`)

### 5.1 Input port

```go
type TranscribeService interface {
    Submit(ctx context.Context, req domain.Request) (Job, error)
    ListProviders() []domain.ProviderID
    ListModels(p domain.ProviderID) ([]string, error)
}

type Job interface {
    ID() string
    Progress() <-chan domain.ProgressEvent   // closed when job ends
    Wait() (*domain.Result, error)           // safe to call repeatedly
    Cancel()                                  // idempotent
}
```

`Submit` returns immediately; the service spawns one goroutine per job. The UIs hold the `Job` and choose how to consume its lifecycle.

### 5.2 Output ports

```go
type Provider interface {
    ID() domain.ProviderID
    MaxUploadBytes() int64                                // 25MB/200MB/1GB depending on API
    Models() []string
    DefaultModel() string
    Capabilities(model string) ModelCapabilities          // validated before pipeline runs
    Transcribe(ctx context.Context, audio domain.AudioFile, opts ProviderOpts) (*domain.Result, error)
}

type ModelCapabilities struct {
    WordTimestamps    bool
    SegmentTimestamps bool
    Diarization       bool         // informational in v1
    LanguageHint      bool
}

type ProviderOpts struct {
    Model    string
    Language string
}

type AudioProcessor interface {
    Probe(path string) (domain.AudioFile, error)
    ExtractAudio(ctx context.Context, videoPath string) (domain.AudioFile, error)
    Transcode(ctx context.Context, in domain.AudioFile, target TargetFormat) (domain.AudioFile, error)
    Chunk(ctx context.Context, in domain.AudioFile, maxBytes int64) ([]domain.Chunk, error)
    Cleanup(f domain.AudioFile)                           // removes IsTemp files
}

type TargetFormat struct {
    Codec      string  // "flac" | "mp3" | "pcm_s16le"
    Bitrate    string  // empty for flac/pcm
    SampleRate int     // 0 = keep source
}

type ConfigStore interface {
    Load() (Config, error)
    Save(Config) error
    Path() string
}

type Config struct {
    APIKeys         map[domain.ProviderID]string
    DefaultProvider domain.ProviderID
    DefaultLanguage string
    FFmpegPath      string         // empty = exec.LookPath("ffmpeg")
}

type ResultCache interface {
    Lookup(inputPath string, p domain.ProviderID) (*domain.Result, bool, error)
    Save(inputPath string, r *domain.Result) error
}

type FormatWriter interface {
    Format() domain.OutputFormat
    Write(r *domain.Result, dst string) error
}

type Logger interface {
    Debug(msg string, kv ...any)
    Info(msg string, kv ...any)
    Warn(msg string, kv ...any)
    Error(msg string, kv ...any)
}
```

**Key invariants:**

- `Provider.Transcribe` receives an `AudioFile` already within `MaxUploadBytes()`. Chunking is service-level, not provider-level.
- `ListProviders()` returns only configured providers (API key non-empty at startup).
- `FormatWriter` is registry-driven: services iterate `Request.Formats` and dispatch by `Format()`. Adding `word_srt` later = one new adapter file.

## 6. Core service (`internal/core/services`)

```
internal/core/services/
├── transcribe.go     # Service struct, Submit, ListProviders, ListModels
├── pipeline.go       # stage machine (one goroutine per Job)
├── chunking.go       # split + merge logic
├── davinci.go        # pause + filler post-processing
└── registry.go       # ProviderID → Provider lookup
```

### 6.1 Service

```go
type Deps struct {
    Providers map[domain.ProviderID]ports.Provider
    Audio     ports.AudioProcessor
    Cache     ports.ResultCache
    Writers   map[domain.OutputFormat]ports.FormatWriter
    Log       ports.Logger
}

type Service struct { deps Deps }
func New(deps Deps) *Service
```

`Service` is stateless apart from its dependencies. `Submit` constructs a `*job`, spawns `go pipeline.Run(j, s)`, returns the handle.

### 6.2 Job

```go
type job struct {
    id       string
    req      domain.Request
    progress chan domain.ProgressEvent  // buffered, cap 32
    done     chan struct{}              // closed on completion
    result   *domain.Result
    err      error
    ctx      context.Context
    cancel   context.CancelFunc
}
```

`Wait` returns the cached `(result, err)` after `done` closes — safe for repeated calls and concurrent waiters.

### 6.3 Pipeline stages

Run inside the job goroutine, in order. Each emits a `ProgressEvent` at entry.

0. **Capability check.** Lookup `provider.Capabilities(model)`. If any `req.Formats` needs timestamps and `caps.WordTimestamps == false`, return `domain.ErrIncompatible` immediately. No file work happens on a misconfigured request.
1. **Probe.** `audio.Probe(req.InputPath)` → metadata.
2. **Cache lookup.** If `req.UseCache`, `cache.Lookup`. Hit → skip to step 7 with the cached `Result`.
3. **Extract.** If input is video, `audio.ExtractAudio` → WAV (tracked `IsTemp`).
4. **Compress.** Pick `TargetFormat` per provider (FLAC for AssemblyAI/ElevenLabs, MP3 128k for Groq/OpenAI), call `audio.Transcode`.
5. **Chunk.** If size > `provider.MaxUploadBytes()`, `audio.Chunk` → `[]Chunk`. Otherwise single-chunk path.
6. **Transcribe.** For each chunk sequentially, `provider.Transcribe(ctx, chunk, opts)`. Merge results: concatenate `Text`, offset `Words/Segments` timestamps by `chunk.StartOffset`, concatenate `RawJSON` into a JSON array. Emits `Percent = i/total` per chunk.
7. **Post-process.** If `DavinciSRT` requested, `davinci.Apply(result, opts)` inserts synthetic `(...)` words for gaps ≥ `SilentPortionThreshold` and tags filler-word matches for the writer.
8. **Cache write.** `cache.Save(req.InputPath, result)`. Skipped on cache-hit path.
9. **Write outputs.** For each `Format`, look up writer, call `Write`. Emits `Percent = i/total`.
10. **Done.** Close `progress` channel.

**Cleanup**: `defer` at the top of `pipeline.Run` collects every `IsTemp` file and calls `audio.Cleanup` even on error / cancel.

**Cancellation**: every blocking call takes the job's `ctx`. `Cancel()` cancels the context; in-flight HTTP requests abort; ffmpeg subprocesses are killed via `exec.Cmd.Cancel`.

**Panic safety**: `pipeline.Run` recovers panics inside the job goroutine and converts them to an error with stack trace logged at error level.

### 6.4 Capability validation rules

- `FormatText` needs no timestamps.
- `FormatSRT` and `FormatDavinciSRT` need `WordTimestamps == true`.
- Future `FormatWordSRT` (v2) will need `WordTimestamps == true`.
- Unknown model → `Capabilities(model)` returns zero-value (all `false`) → only `Text` is allowed. Fail-safe.

## 7. Delivery layer

### 7.1 Composition root (`internal/delivery/wire.go`)

One function called by every UI:

```go
func BuildService(cfg ports.Config, log ports.Logger) (ports.TranscribeService, error)
```

Instantiates the ffmpeg adapter (resolving `cfg.FFmpegPath` → `exec.LookPath("ffmpeg")` → `ErrFFmpegMissing`), iterates `cfg.APIKeys` and wires only the providers with non-empty keys, builds the format-writer map, returns a `*services.Service`. UIs never import an adapter directly.

### 7.2 Mode selection (`cmd/transcribe/main.go`)

`decideUIMode(os.Args)` in priority order:

1. `--json` → `ModeJSON` (and reject `--ui` if combined — exit 2).
2. `--ui=<x>` → that mode.
3. Zero args → `ModeGUI` (v1 is Windows-only; a desktop session is always present). v2 adds a `runtime.GOOS == "linux" && os.Getenv("DISPLAY") == ""` check that falls back to `ModeTUI` with a log line.
4. Some args but missing required ones (file + provider) → `ModeTUI`, prefilled with the flags that were given.
5. All required args present → `ModeCLI`.

### 7.3 CLI (`internal/delivery/cli/`)

Cobra. Root command `transcribe`, positional `<file...>`, flags mirror the Python surface: `--api`, `--model`, `--language`, `--output`, `--output-dir`, `--use-cache`, `--silent-portion`, `--davinci`, `--ffmpeg-path`, etc. Subcommands: `setup` (non-interactive config writer), `providers` (lists configured providers and models — JSON-aware).

Flow:

```go
job, err := svc.Submit(ctx, req)
if err != nil { renderError(err); os.Exit(exitCode(err)) }

go renderProgress(job.Progress())  // stderr text, or stdout JSONL if --json --progress

result, err := job.Wait()
renderResult(result, err)
os.Exit(exitCode(err))
```

Signals: `SIGINT`/`SIGTERM` → `job.Cancel()` → wait ≤ 5 s → exit `130`.

#### 7.3.1 Agent-callable `--json` mode

`--json` forces three behaviors:

1. **Pure CLI, no escalation.** Missing required input → exit 2, no TUI ever opens. `--json` + `--ui=<x>` is rejected.
2. **Stdout is JSON.** All non-JSON diagnostics go to stderr. Optional `--json-logs` puts stderr output as JSON Lines too.
3. **Deterministic schema** in `internal/delivery/cli/json_render.go`, the single source of truth:
   - Without `--progress`: one final object — `{"schema_version":1, "status":"ok"|"error", "result"?:{...}, "error"?:{...}}`.
   - With `--progress`: JSON Lines stream — `{"type":"progress", "stage":"...", "percent":0.5, ...}` per event, terminated by `{"type":"result", ...}` or `{"type":"error", ...}`.

#### 7.3.2 Exit codes

- `0` success
- `1` internal error (unexpected panic, bug)
- `2` usage error (bad flags, missing required input, capability mismatch)
- `3` config error (missing API key, missing ffmpeg)
- `4` provider error (HTTP failure after retries, auth rejected)
- `5` audio/ffmpeg error
- `130` canceled via signal

These match the `"code"` field in JSON error payloads so an agent can branch on either.

### 7.4 TUI (`internal/delivery/tui/`)

Bubble Tea, three screens (file picker, provider/options, progress) in a single `App` model state machine. Launched from a Cobra command (`tui.Run(svc, cfg)`). Prefilled flags from CLI escalation become the initial model state.

Progress events from `job.Progress()` get bridged via a `tea.Cmd` that reads one event and returns a `progressMsg`; `Update` re-issues the `Cmd` until the channel closes. `q`/`Esc` calls `job.Cancel()` and waits for the channel to close before exiting.

### 7.5 GUI (`internal/delivery/gui/`)

Fyne. One main window with file drop zone + browse button (`dialog.ShowFileOpen`), provider dropdown (`svc.ListProviders()`), model dropdown (`svc.ListModels(provider)` on change), format checkboxes, language entry, Start button, progress widget, log pane. Settings menu opens a separate window editing the config file.

Threading: Start spawns a goroutine that ranges over `job.Progress()` and marshals UI updates via `fyne.Do`. Progress widget switches between determinate (`Percent >= 0`) and indeterminate (`Percent < 0`).

Window close mid-job: confirm dialog → `job.Cancel()` → wait → close.

## 8. Adapters

### 8.1 Provider clients (`internal/adapters/api/<name>/`)

Identical shape per provider:

- `client.go` — `New(apiKey, *http.Client) *Client` implementing `ports.Provider`. Plain `net/http`; no vendor SDKs unless one is exemplary. All requests honor `ctx`.
- `models.go` — static `modelCaps` map + `DefaultModel()`.
- `parse.go` — pure function converting provider JSON → `domain.Result`. Unit-tested against fixtures lifted from the Python `test/sample_*.json`.

Retry policy in `internal/adapters/api/internal/retry/`: `Do(ctx, attempts=3, base=5s, fn)` with jittered backoff and an `IsRetryable(err)` classifier (5xx, 429, timeouts → retry; auth/4xx → don't).

### 8.2 FFmpeg (`internal/adapters/audio/`)

`New(ffmpegPath, ffprobePath string, log ports.Logger) (*FFmpeg, error)` — empty paths fall through to `exec.LookPath`; both-missing returns `ErrFFmpegMissing` from `BuildService`, before any UI launches.

- `Probe` shells `ffprobe -v error -show_streams -show_format -of json`.
- `ExtractAudio` shells `ffmpeg -i <in> -vn -acodec pcm_s16le -ac 1 -ar 16000 <tmp.wav>`.
- `Transcode` builds args from `TargetFormat`.
- `Chunk` computes per-chunk duration to stay under `maxBytes`, then `ffmpeg -ss <off> -t <dur> -c copy`.
- Temp files in `os.TempDir()/transcribe-<jobid>/`, tracked by `IsTemp`.
- All `exec.Cmd` honor `ctx` — cancellation kills subprocesses.

### 8.3 Config (`internal/adapters/config/`)

TOML via `pelletier/go-toml/v2`. Location:

```
%LOCALAPPDATA%\transcribe\config.toml   (Windows — v1)
~/.transcribe/config.toml               (Unix — v2)
```

Schema:

```toml
default_provider = "groq"
default_language = "en"
ffmpeg_path = ""

[api_keys]
groq = "gsk_..."
openai = "sk-..."
elevenlabs = ""
```

Environment variables override file values: `TRANSCRIBE_GROQ_KEY`, `TRANSCRIBE_OPENAI_KEY`, `TRANSCRIBE_ASSEMBLYAI_KEY`, `TRANSCRIBE_ELEVENLABS_KEY`, `TRANSCRIBE_GEMINI_KEY`, `TRANSCRIBE_MISTRAL_KEY`, and `TRANSCRIBE_FFMPEG_PATH`. Useful for CI and agent invocations.

### 8.4 Cache (`internal/adapters/cache/`)

Sidecar file `<input-basename>.transcribe.<provider>.json` next to the input. Versioned envelope:

```json
{
  "schema_version": 1,
  "provider": "groq",
  "model": "whisper-large-v3",
  "language": "en",
  "duration_ms": 192340,
  "text": "…",
  "words": [...],
  "segments": [...],
  "raw": {…provider response…}
}
```

Unknown `schema_version` → cache miss + warning log.

### 8.5 Format writers (`internal/adapters/format/`)

Three implementations:

- `text.go` — `r.Text` to disk.
- `srt.go` — groups `r.Words` into subtitle blocks (≤ ~7 words or ~3 s per block, configurable via constants) with standard timecodes.
- `davinci.go` — same grouping but reads the post-processing markers from `r.Words` (synthetic `(...)` and filler flags) and emits the DaVinci-flavored output. Shares the grouping helper with `srt.go`.

### 8.6 Logging (`internal/adapters/logging/`)

`log/slog` (Go stdlib) wrapped in `ports.Logger`. CLI sets a text handler on stderr; TUI swaps in a handler that pushes log lines into a `tea.Msg`; GUI uses a handler that appends to a Fyne text widget.

## 9. Testing

Three layers, `testing` + `testify/require`.

```
internal/.../*_test.go        # unit tests per package
tests/integration/            # provider-hitting tests, build tag "integration"
testdata/                     # audio fixtures + golden outputs + provider JSON samples
```

- **Unit tests** for each `adapters/api/*/parse.go` against fixtures from the Python `test/sample_*.json`. One golden test per format writer — byte-for-byte compare against a checked-in SRT/text file.
- **Service tests** with a fake `Provider` + fake `AudioProcessor`, table-driven over the pipeline state machine: cache hit/miss, chunking branch, video extraction branch, cancellation mid-stage, capability rejection.
- **Integration tests** under `tests/integration/` with build tag `integration`. Each test calls `t.Skip` when its env key is absent. Each provider gets one happy-path test against a ~5 s audio fixture.
- **CLI-level test** spawns the built binary with `--json`, feeds a known input, asserts the JSON schema with `encoding/json` round-trip + field checks. Pins the agent contract.
- **Race detector** in CI: `go test -race ./...`.

## 10. Distribution

### v1: Windows-only

- Single binary `transcribe-windows-amd64.exe`.
- Optional `transcribe-gui-windows-amd64.exe` built with `-ldflags="-H windowsgui"` so Explorer launches don't pop a console window.
- Local dev on Windows 11 with `go build`/`go run` against the installed Go toolchain. Fyne's CGO needs the MinGW-w64 gcc that comes with the Fyne setup docs; one-time install.
- `go build -tags nogui ./cmd/transcribe` produces a Fyne-less binary for fast CLI/TUI iteration without dragging in OpenGL deps.
- Version injection: `-ldflags="-X main.version=v1.0.0"`. `transcribe --version` reports it.

### v1 scaffolding for v2 cross-platform

- Build tags `//go:build !nogui` on every Fyne import so headless builds are clean.
- `internal/adapters/config` already chooses path by `runtime.GOOS`.
- Test suite runs on Linux today (CLI/TUI subset) — Fyne tests get a `//go:build !linux || cgo` guard.
- No GoReleaser config yet; one-pager `BUILD.md` documents the manual Windows build until v2.

### v2 backlog (not in this spec)

- macOS amd64 + arm64 builds, macOS signing/notarization (needs Apple Developer cert).
- Linux amd64 build.
- GoReleaser pipeline + GitHub Actions matrix.
- Speaker diarization (ElevenLabs `--diarize`).
- Word-level SRT (`word_srt`).
- Drag-and-drop batch templates.

## 11. Migration

Python source remains untouched on `main` during the porting period. Once Go reaches the v1 parity bar (this spec), a single follow-up PR:

1. Removes `audio_transcribe/`, `transcribe.py`, `build.py`, `transcribe.spec`, `transcribe-windows-amd64.spec`, the Python-era `*.log` files in the repo root, `test_all_apis.py`, the Python `test/`, the Python `tests/`.
2. Rewrites `README.md` and `CLAUDE.md` for the Go codebase.
3. Updates the GitHub release workflow if any.

The Python tool's behavior is the v1 acceptance bar: same outputs given the same inputs, modulo the deliberate departures called out in section 4 (`time.Duration` timestamps, schema-versioned cache JSON, TOML config replacing `.env`).

## 12. Constraints (from the brief)

- [ ] No `fmt.Print` inside `internal/core` or `internal/adapters`. Errors and data flow via return values and the `ports.Logger`.
- [ ] Job/progress concurrency via goroutines + channels (channel-on-Job-handle model).
- [ ] Single cohesive executable per platform.

## 13. Open questions

None blocking — all major design choices answered during brainstorming. Implementation-level decisions (specific SRT block grouping thresholds, exact retry backoff curve, exact ffmpeg arg sets) get nailed down in the implementation plan.
