# Go Port — Plan 1: Foundation + Groq + CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a working `transcribe.exe` that does file → Groq → `.txt`/`.srt`/`.davinci.srt` via the Cobra CLI (including `--json` agent mode), with the hexagonal foundation in place so Plans 2-4 can layer providers and UIs on top.

**Architecture:** Hexagonal (ports & adapters) per `docs/superpowers/specs/2026-05-15-go-port-design.md`. Single Go module, single binary, dev on Windows 11.

**Tech Stack:** Go 1.22+, Cobra, `pelletier/go-toml/v2`, `log/slog`, `testify/require`. FFmpeg shelled via `os/exec`. Bubble Tea/Fyne arrive in later plans.

**Prerequisite (execution-time, not part of this plan):** Create a worktree via the `superpowers:using-git-worktrees` skill (e.g. branch `go-port-foundation`). All commits in this plan land on that branch. The Python tree on `main` stays untouched.

---

## File map

```
go.mod
.gitignore
BUILD.md

internal/core/domain/
├── ids.go              # ProviderID, OutputFormat + constants
├── transcription.go    # Request, Result, Word, Segment, Speaker, DaVinciOptions
├── audio.go            # AudioFile, AudioFormat, Chunk
├── progress.go         # ProgressEvent, Stage
└── errors.go           # sentinels + ErrIncompatible + ErrProvider

internal/ports/
├── service.go          # TranscribeService, Job
├── provider.go         # Provider, ModelCapabilities, ProviderOpts
├── audio.go            # AudioProcessor, TargetFormat
├── config.go           # ConfigStore, Config
├── cache.go            # ResultCache
├── format.go           # FormatWriter
└── logger.go           # Logger

internal/adapters/config/
├── tomlstore.go
└── tomlstore_test.go

internal/adapters/cache/
├── sidecar.go
└── sidecar_test.go

internal/adapters/format/
├── grouping.go         # word→subtitle-block grouper, shared by srt + davinci
├── text.go
├── srt.go
├── davinci.go
├── grouping_test.go
├── text_test.go
├── srt_test.go
└── davinci_test.go

internal/adapters/audio/
├── ffmpeg.go           # type, constructor, helpers
├── atomic.go           # *.partial → final atomic rename
├── meta.go             # meta.json read/write
├── probe.go
├── copy.go             # CopyAudio
├── extract.go          # ExtractAudio
├── transcode.go
├── chunk.go
├── cleanup.go
└── *_test.go           # one per source file, skip if ffmpeg missing

internal/adapters/api/internal/retry/
├── retry.go
└── retry_test.go

internal/adapters/api/groq/
├── client.go
├── models.go
├── parse.go
├── parse_test.go
└── integration_test.go    # build tag: integration

internal/adapters/logging/
└── slog.go

internal/core/services/
├── registry.go
├── service.go
├── job.go
├── transient.go        # transient(err) classifier
├── prepare.go          # copy-first decision tree (pipeline step 5)
├── pipeline.go         # pipeline.Run state machine
├── chunking.go         # multi-chunk result merge
├── davinci.go          # pause + filler post-processing
└── *_test.go

internal/delivery/
└── wire.go             # BuildService composition root

internal/delivery/cli/
├── root.go             # cobra root + version
├── transcribe.go       # transcribe command + run
├── setup.go            # transcribe setup subcommand
├── providers.go        # transcribe providers subcommand
├── json_render.go      # event/result/error renderers
├── exit.go             # exitCode(err) → POSIX code
├── signal.go           # SIGINT/SIGTERM → cancel
└── render_test.go

cmd/transcribe/
└── main.go             # decideUIMode + dispatch

testdata/
├── short-sample.mp3
└── groq_sample.json

tests/integration/
└── cli_test.go         # built-binary smoke (build tag: integration)
```

---

## Phase A — Project skeleton

### Task A1: Initialize the Go module and ignore files

**Files:**

- Create: `go.mod`
- Create: `.gitignore`

- [ ] **Step 1: Initialize the module**

Run from worktree root:

```bash
go mod init github.com/leotulipan/transcribe
```

Expected: `go.mod` written with the module path.

- [ ] **Step 2: Write `.gitignore`**

```gitignore
# Build artifacts
/dist/
/bin/
*.exe
transcribe
transcribe-gui

# Test artifacts
*.test
*.out
coverage.out

# IDE / OS
.vscode/
.idea/
.DS_Store
Thumbs.db

# Transcribe runtime
*.transcribe.*.json
.transcribe-tmp/
```

- [ ] **Step 3: Add the testify dependency**

```bash
go get github.com/stretchr/testify@latest
```

Expected: `go.sum` appears, `go.mod` gains a `require` line.

- [ ] **Step 4: Commit**

```bash
git add go.mod go.sum .gitignore
git commit -m "chore: init go module"
```

---

### Task A2: BUILD.md with Windows instructions

**Files:**

- Create: `BUILD.md`

- [ ] **Step 1: Write `BUILD.md`**

```markdown
# Build

## Prerequisites (Windows 11)

- Go 1.22 or newer — https://go.dev/dl/
- ffmpeg + ffprobe on PATH (`winget install Gyan.FFmpeg` or `choco install ffmpeg`)

## Build

```powershell
go build -ldflags "-X main.version=dev" -o bin/transcribe.exe ./cmd/transcribe
```

Release build with embedded version:

```powershell
$ver = (git describe --tags --always)
go build -ldflags "-X main.version=$ver" -o bin/transcribe.exe ./cmd/transcribe
```

## Run tests

```powershell
go test ./...

# With race detector (slower)
go test -race ./...

# Integration tests (skipped unless API keys present)
$env:TRANSCRIBE_GROQ_KEY = "gsk_..."
go test -tags integration ./tests/integration/...
```

## Notes

- v1 ships Windows-only. macOS/Linux are v2 and require additional toolchain
  setup (Fyne CGO + macOS signing).
- The TUI/GUI delivery layers are not yet wired (Plans 3 and 4).

```

- [ ] **Step 2: Commit**

```bash
git add BUILD.md
git commit -m "docs: build instructions for windows"
```

---

## Phase B — Domain types

### Task B1: ProviderID, OutputFormat, and their constants

**Files:**

- Create: `internal/core/domain/ids.go`

- [ ] **Step 1: Write `internal/core/domain/ids.go`**

```go
package domain

type ProviderID string

const (
    ProviderAssemblyAI ProviderID = "assemblyai"
    ProviderElevenLabs ProviderID = "elevenlabs"
    ProviderGroq       ProviderID = "groq"
    ProviderOpenAI     ProviderID = "openai"
    ProviderGemini     ProviderID = "gemini"
    ProviderMistral    ProviderID = "mistral"
)

type OutputFormat string

const (
    FormatText       OutputFormat = "text"
    FormatSRT        OutputFormat = "srt"
    FormatDavinciSRT OutputFormat = "davinci_srt"
)

// NeedsTimestamps reports whether this output format requires word-level timing.
func (f OutputFormat) NeedsTimestamps() bool {
    return f == FormatSRT || f == FormatDavinciSRT
}
```

- [ ] **Step 2: Build to verify it compiles**

```bash
go build ./internal/core/domain/...
```

Expected: no output (success).

- [ ] **Step 3: Commit**

```bash
git add internal/core/domain/ids.go
git commit -m "feat(domain): provider and output format ids"
```

---

### Task B2: Transcription value types

**Files:**

- Create: `internal/core/domain/transcription.go`

- [ ] **Step 1: Write `internal/core/domain/transcription.go`**

```go
package domain

import "time"

// Request describes a single transcription job submitted to the service.
type Request struct {
    InputPath   string
    Provider    ProviderID
    Model       string         // "" = provider default
    Language    string         // ISO-639-1; "" = auto-detect
    Formats     []OutputFormat
    OutputDir   string         // "" = next to input
    DaVinciOpts *DaVinciOptions
    UseCache    bool
}

// Result is the normalized output every provider produces.
type Result struct {
    Provider   ProviderID
    Model      string
    Language   string
    Text       string
    Confidence float64
    Words      []Word
    Segments   []Segment
    Speakers   []Speaker      // empty in v1
    Duration   time.Duration
    SourcePath string
    RawJSON    []byte         // pristine provider response (JSON array if merged)
}

type Word struct {
    Text       string
    Start, End time.Duration
    Confidence float64
}

type Segment struct {
    Text       string
    Start, End time.Duration
    SpeakerID  string
}

type Speaker struct {
    ID, Label string
}

type DaVinciOptions struct {
    SilentPortionThreshold time.Duration
    PaddingStart           time.Duration
    FillerWords            []string
}

// DefaultFillerWords is what DaVinciOptions.FillerWords defaults to when empty.
var DefaultFillerWords = []string{"um", "uh", "ähm", "äh", "hm", "hmm"}
```

- [ ] **Step 2: Build**

```bash
go build ./internal/core/domain/...
```

- [ ] **Step 3: Commit**

```bash
git add internal/core/domain/transcription.go
git commit -m "feat(domain): request/result value types"
```

---

### Task B3: Audio value types

**Files:**

- Create: `internal/core/domain/audio.go`

- [ ] **Step 1: Write `internal/core/domain/audio.go`**

```go
package domain

import "time"

// AudioFile describes an audio (or audio-bearing) file on disk.
type AudioFile struct {
    Path      string
    SizeBytes int64
    Duration  time.Duration
    Container string  // mp4, m4a, wav, mp3, flac, ogg, webm
    Codec     string  // aac, mp3, pcm_s16le, flac, opus
    IsTemp    bool    // managed temp file; cleanup eligible
    Complete  bool    // ffmpeg returned 0 and file fully on disk
    Chunks    []Chunk
}

// AudioFormat describes an accepted container/codec combination.
// An empty Container means "any container is fine as long as Codec matches".
type AudioFormat struct {
    Container string
    Codec     string
}

// Chunk is one slice of a larger audio file produced for size-limited APIs.
type Chunk struct {
    Path        string
    StartOffset time.Duration
    SizeBytes   int64
    Complete    bool
}
```

- [ ] **Step 2: Build**

```bash
go build ./internal/core/domain/...
```

- [ ] **Step 3: Commit**

```bash
git add internal/core/domain/audio.go
git commit -m "feat(domain): audio value types"
```

---

### Task B4: Progress and stage enum

**Files:**

- Create: `internal/core/domain/progress.go`

- [ ] **Step 1: Write `internal/core/domain/progress.go`**

```go
package domain

import "time"

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

func (s Stage) String() string {
    switch s {
    case StageProbing:      return "probing"
    case StageExtracting:   return "extracting"
    case StageCompressing:  return "compressing"
    case StageChunking:     return "chunking"
    case StageUploading:    return "uploading"
    case StageTranscribing: return "transcribing"
    case StageParsing:      return "parsing"
    case StageWriting:      return "writing"
    case StageDone:         return "done"
    }
    return "unknown"
}

// ProgressEvent flows from the service to UIs over Job.Progress().
type ProgressEvent struct {
    Stage   Stage
    Message string
    Percent float64        // -1 when not estimable
    Elapsed time.Duration
}
```

- [ ] **Step 2: Build**

```bash
go build ./internal/core/domain/...
```

- [ ] **Step 3: Commit**

```bash
git add internal/core/domain/progress.go
git commit -m "feat(domain): progress events and stage enum"
```

---

### Task B5: Domain error types

**Files:**

- Create: `internal/core/domain/errors.go`
- Create: `internal/core/domain/errors_test.go`

- [ ] **Step 1: Write the failing test**

```go
package domain

import (
    "errors"
    "testing"

    "github.com/stretchr/testify/require"
)

func TestErrIncompatible_Error(t *testing.T) {
    e := ErrIncompatible{
        Provider: ProviderGroq,
        Model:    "whisper-large-v3",
        Format:   FormatSRT,
        Reason:   "model returns text only",
    }
    msg := e.Error()
    require.Contains(t, msg, "groq")
    require.Contains(t, msg, "whisper-large-v3")
    require.Contains(t, msg, "srt")
    require.Contains(t, msg, "text only")
}

func TestErrProvider_Unwrap(t *testing.T) {
    base := errors.New("boom")
    e := &ErrProvider{Provider: ProviderGroq, StatusCode: 500, Retryable: true, Cause: base}
    require.True(t, errors.Is(e, base))
}
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
go test ./internal/core/domain/...
```

Expected: compile error (`ErrIncompatible` undefined).

- [ ] **Step 3: Write `internal/core/domain/errors.go`**

```go
package domain

import (
    "errors"
    "fmt"
)

var (
    ErrConfigMissing   = errors.New("config error")
    ErrProviderMissing = errors.New("provider not configured")
    ErrFFmpegMissing   = errors.New("ffmpeg not found")
    ErrCanceled        = errors.New("canceled")
)

// ErrIncompatible signals a request/model/format mismatch caught before any
// expensive work runs.
type ErrIncompatible struct {
    Provider ProviderID
    Model    string
    Format   OutputFormat
    Reason   string
}

func (e ErrIncompatible) Error() string {
    return fmt.Sprintf("incompatible: %s/%s cannot produce %s — %s",
        e.Provider, e.Model, e.Format, e.Reason)
}

// ErrProvider wraps an upstream API failure with classification hints.
type ErrProvider struct {
    Provider   ProviderID
    StatusCode int
    Retryable  bool
    Cause      error
}

func (e *ErrProvider) Error() string {
    if e.StatusCode == 0 {
        return fmt.Sprintf("%s: %v", e.Provider, e.Cause)
    }
    return fmt.Sprintf("%s: http %d: %v", e.Provider, e.StatusCode, e.Cause)
}

func (e *ErrProvider) Unwrap() error { return e.Cause }
```

- [ ] **Step 4: Run the tests**

```bash
go test ./internal/core/domain/...
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add internal/core/domain/errors.go internal/core/domain/errors_test.go
git commit -m "feat(domain): typed errors"
```

---

## Phase C — Ports

### Task C1: Input port (TranscribeService, Job)

**Files:**

- Create: `internal/ports/service.go`

- [ ] **Step 1: Write `internal/ports/service.go`**

```go
package ports

import (
    "context"

    "github.com/leotulipan/transcribe/internal/core/domain"
)

// TranscribeService is the input port the delivery layer calls.
type TranscribeService interface {
    // Submit kicks off a transcription on a background goroutine and returns a
    // Job handle. The service owns the goroutine; the caller owns the handle.
    Submit(ctx context.Context, req domain.Request) (Job, error)

    // ListProviders returns providers configured at startup (API key present).
    ListProviders() []domain.ProviderID

    // ListModels returns the model names a provider advertises.
    ListModels(p domain.ProviderID) ([]string, error)
}

// Job is a handle to an in-flight (or finished) transcription.
type Job interface {
    ID() string
    Progress() <-chan domain.ProgressEvent // closed when the job ends
    Wait() (*domain.Result, error)         // blocks until done; safe for repeated calls
    Cancel()                                // idempotent
}
```

- [ ] **Step 2: Build**

```bash
go build ./internal/ports/...
```

- [ ] **Step 3: Commit**

```bash
git add internal/ports/service.go
git commit -m "feat(ports): input port (service + job)"
```

---

### Task C2: Provider output port

**Files:**

- Create: `internal/ports/provider.go`

- [ ] **Step 1: Write `internal/ports/provider.go`**

```go
package ports

import (
    "context"

    "github.com/leotulipan/transcribe/internal/core/domain"
)

// Provider is implemented by each transcription API adapter.
type Provider interface {
    ID() domain.ProviderID
    MaxUploadBytes() int64
    Models() []string
    DefaultModel() string

    // Capabilities returns model-level capabilities. Service validates them
    // against Request.Formats before any audio work begins.
    Capabilities(model string) ModelCapabilities

    // Transcribe ingests a file already within MaxUploadBytes() and an accepted
    // codec, returns a normalized Result.
    Transcribe(ctx context.Context, audio domain.AudioFile, opts ProviderOpts) (*domain.Result, error)
}

type ModelCapabilities struct {
    WordTimestamps    bool
    SegmentTimestamps bool
    Diarization       bool                  // informational in v1
    LanguageHint      bool
    AcceptedInputs    []domain.AudioFormat
}

type ProviderOpts struct {
    Model    string
    Language string
}
```

- [ ] **Step 2: Build**

```bash
go build ./internal/ports/...
```

- [ ] **Step 3: Commit**

```bash
git add internal/ports/provider.go
git commit -m "feat(ports): provider port + model capabilities"
```

---

### Task C3: Audio processor output port

**Files:**

- Create: `internal/ports/audio.go`

- [ ] **Step 1: Write `internal/ports/audio.go`**

```go
package ports

import (
    "context"

    "github.com/leotulipan/transcribe/internal/core/domain"
)

// AudioProcessor wraps ffmpeg/ffprobe operations.
type AudioProcessor interface {
    Probe(path string) (domain.AudioFile, error)

    // CopyAudio stream-copies the audio track into a derived container without
    // re-encoding. workDir is the directory in which to land the output.
    CopyAudio(ctx context.Context, in domain.AudioFile, workDir string) (domain.AudioFile, error)

    // ExtractAudio decodes to 16-bit mono PCM WAV — the lossless fallback when
    // stream copy isn't viable.
    ExtractAudio(ctx context.Context, videoPath string, workDir string) (domain.AudioFile, error)

    // Transcode re-encodes to a target codec/bitrate.
    Transcode(ctx context.Context, in domain.AudioFile, target TargetFormat, workDir string) (domain.AudioFile, error)

    // Chunk slices a file into byte-size-bounded chunks.
    Chunk(ctx context.Context, in domain.AudioFile, maxBytes int64, workDir string) ([]domain.Chunk, error)

    // Cleanup removes one tempfile. Caller decides when based on the cleanup
    // policy in services.
    Cleanup(f domain.AudioFile) error
}

type TargetFormat struct {
    Codec      string // "flac" | "mp3" | "pcm_s16le"
    Bitrate    string // empty for flac/pcm
    SampleRate int    // 0 = keep source
}
```

- [ ] **Step 2: Build**

```bash
go build ./internal/ports/...
```

- [ ] **Step 3: Commit**

```bash
git add internal/ports/audio.go
git commit -m "feat(ports): audio processor port"
```

---

### Task C4: Config, cache, format, and logger ports

**Files:**

- Create: `internal/ports/config.go`
- Create: `internal/ports/cache.go`
- Create: `internal/ports/format.go`
- Create: `internal/ports/logger.go`

- [ ] **Step 1: Write `internal/ports/config.go`**

```go
package ports

import "github.com/leotulipan/transcribe/internal/core/domain"

type ConfigStore interface {
    Load() (Config, error)
    Save(Config) error
    Path() string
}

type Config struct {
    APIKeys         map[domain.ProviderID]string
    DefaultProvider domain.ProviderID
    DefaultLanguage string
    FFmpegPath      string // empty = exec.LookPath("ffmpeg")
}
```

- [ ] **Step 2: Write `internal/ports/cache.go`**

```go
package ports

import "github.com/leotulipan/transcribe/internal/core/domain"

type ResultCache interface {
    Lookup(inputPath string, p domain.ProviderID) (*domain.Result, bool, error)
    Save(inputPath string, r *domain.Result) error
}
```

- [ ] **Step 3: Write `internal/ports/format.go`**

```go
package ports

import "github.com/leotulipan/transcribe/internal/core/domain"

type FormatWriter interface {
    Format() domain.OutputFormat
    Write(r *domain.Result, dst string) error
}
```

- [ ] **Step 4: Write `internal/ports/logger.go`**

```go
package ports

type Logger interface {
    Debug(msg string, kv ...any)
    Info(msg string, kv ...any)
    Warn(msg string, kv ...any)
    Error(msg string, kv ...any)
}
```

- [ ] **Step 5: Build**

```bash
go build ./internal/ports/...
```

- [ ] **Step 6: Commit**

```bash
git add internal/ports/config.go internal/ports/cache.go internal/ports/format.go internal/ports/logger.go
git commit -m "feat(ports): config/cache/format/logger output ports"
```

---

## Phase D — Config adapter

### Task D1: TOML config store + env override

**Files:**

- Create: `internal/adapters/config/tomlstore.go`
- Create: `internal/adapters/config/tomlstore_test.go`

- [ ] **Step 1: Add the toml dependency**

```bash
go get github.com/pelletier/go-toml/v2@latest
```

- [ ] **Step 2: Write the failing test**

```go
package config

import (
    "os"
    "path/filepath"
    "testing"

    "github.com/stretchr/testify/require"

    "github.com/leotulipan/transcribe/internal/core/domain"
)

func TestStore_RoundTrip(t *testing.T) {
    dir := t.TempDir()
    s := newWithPath(filepath.Join(dir, "config.toml"))

    in := ports_Config{
        APIKeys: map[domain.ProviderID]string{
            domain.ProviderGroq: "gsk_xyz",
        },
        DefaultProvider: domain.ProviderGroq,
        DefaultLanguage: "en",
        FFmpegPath:      `C:\tools\ffmpeg.exe`,
    }
    require.NoError(t, s.Save(in))

    out, err := s.Load()
    require.NoError(t, err)
    require.Equal(t, in.APIKeys[domain.ProviderGroq], out.APIKeys[domain.ProviderGroq])
    require.Equal(t, in.DefaultProvider, out.DefaultProvider)
    require.Equal(t, in.DefaultLanguage, out.DefaultLanguage)
    require.Equal(t, in.FFmpegPath, out.FFmpegPath)
}

func TestStore_EnvOverride(t *testing.T) {
    dir := t.TempDir()
    s := newWithPath(filepath.Join(dir, "config.toml"))
    require.NoError(t, s.Save(ports_Config{
        APIKeys: map[domain.ProviderID]string{domain.ProviderGroq: "from_file"},
    }))

    t.Setenv("TRANSCRIBE_GROQ_KEY", "from_env")
    t.Setenv("TRANSCRIBE_FFMPEG_PATH", `C:\override\ffmpeg.exe`)

    out, err := s.Load()
    require.NoError(t, err)
    require.Equal(t, "from_env", out.APIKeys[domain.ProviderGroq])
    require.Equal(t, `C:\override\ffmpeg.exe`, out.FFmpegPath)
}

func TestStore_LoadMissingFileReturnsEmpty(t *testing.T) {
    s := newWithPath(filepath.Join(t.TempDir(), "missing.toml"))
    out, err := s.Load()
    require.NoError(t, err)
    require.NotNil(t, out.APIKeys)
    require.Empty(t, out.APIKeys)
}

// Local alias so the test file doesn't import the ports package — we just need
// the same struct shape for round-trip checks.
type ports_Config = struct {
    APIKeys         map[domain.ProviderID]string
    DefaultProvider domain.ProviderID
    DefaultLanguage string
    FFmpegPath      string
}

// Test the OS-specific default path returns something non-empty.
func TestDefaultPath_NotEmpty(t *testing.T) {
    require.NotEmpty(t, defaultPath())
    _ = os.Getenv("LOCALAPPDATA") // touch to placate linters
}
```

- [ ] **Step 3: Run the test to confirm it fails**

```bash
go test ./internal/adapters/config/...
```

Expected: compile error (`newWithPath`, `defaultPath` undefined).

- [ ] **Step 4: Write `internal/adapters/config/tomlstore.go`**

```go
package config

import (
    "errors"
    "io/fs"
    "os"
    "path/filepath"
    "runtime"

    "github.com/pelletier/go-toml/v2"

    "github.com/leotulipan/transcribe/internal/core/domain"
    "github.com/leotulipan/transcribe/internal/ports"
)

// envKeys maps provider IDs to environment variable names.
var envKeys = map[domain.ProviderID]string{
    domain.ProviderAssemblyAI: "TRANSCRIBE_ASSEMBLYAI_KEY",
    domain.ProviderElevenLabs: "TRANSCRIBE_ELEVENLABS_KEY",
    domain.ProviderGroq:       "TRANSCRIBE_GROQ_KEY",
    domain.ProviderOpenAI:     "TRANSCRIBE_OPENAI_KEY",
    domain.ProviderGemini:     "TRANSCRIBE_GEMINI_KEY",
    domain.ProviderMistral:    "TRANSCRIBE_MISTRAL_KEY",
}

const envFFmpegPath = "TRANSCRIBE_FFMPEG_PATH"

// fileShape is the on-disk TOML schema.
type fileShape struct {
    DefaultProvider string            `toml:"default_provider"`
    DefaultLanguage string            `toml:"default_language"`
    FFmpegPath      string            `toml:"ffmpeg_path"`
    APIKeys         map[string]string `toml:"api_keys"`
}

type Store struct {
    path string
}

// New returns a Store using the OS-default path.
func New() *Store {
    return newWithPath(defaultPath())
}

func newWithPath(p string) *Store {
    return &Store{path: p}
}

func defaultPath() string {
    if runtime.GOOS == "windows" {
        base := os.Getenv("LOCALAPPDATA")
        if base == "" {
            base = filepath.Join(os.Getenv("USERPROFILE"), "AppData", "Local")
        }
        return filepath.Join(base, "transcribe", "config.toml")
    }
    home, _ := os.UserHomeDir()
    return filepath.Join(home, ".transcribe", "config.toml")
}

func (s *Store) Path() string { return s.path }

func (s *Store) Load() (ports.Config, error) {
    cfg := ports.Config{APIKeys: map[domain.ProviderID]string{}}

    data, err := os.ReadFile(s.path)
    switch {
    case errors.Is(err, fs.ErrNotExist):
        // OK, empty config
    case err != nil:
        return cfg, err
    default:
        var fs_ fileShape
        if err := toml.Unmarshal(data, &fs_); err != nil {
            return cfg, err
        }
        cfg.DefaultProvider = domain.ProviderID(fs_.DefaultProvider)
        cfg.DefaultLanguage = fs_.DefaultLanguage
        cfg.FFmpegPath = fs_.FFmpegPath
        for k, v := range fs_.APIKeys {
            cfg.APIKeys[domain.ProviderID(k)] = v
        }
    }

    // Env overrides
    for id, env := range envKeys {
        if v := os.Getenv(env); v != "" {
            cfg.APIKeys[id] = v
        }
    }
    if v := os.Getenv(envFFmpegPath); v != "" {
        cfg.FFmpegPath = v
    }
    return cfg, nil
}

func (s *Store) Save(cfg ports.Config) error {
    fs_ := fileShape{
        DefaultProvider: string(cfg.DefaultProvider),
        DefaultLanguage: cfg.DefaultLanguage,
        FFmpegPath:      cfg.FFmpegPath,
        APIKeys:         map[string]string{},
    }
    for k, v := range cfg.APIKeys {
        fs_.APIKeys[string(k)] = v
    }

    data, err := toml.Marshal(fs_)
    if err != nil {
        return err
    }
    if err := os.MkdirAll(filepath.Dir(s.path), 0o700); err != nil {
        return err
    }
    return os.WriteFile(s.path, data, 0o600)
}
```

- [ ] **Step 5: Run the tests to confirm they pass**

```bash
go test ./internal/adapters/config/...
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add internal/adapters/config/
git commit -m "feat(config): toml store with env override"
```

---

## Phase E — Cache adapter

### Task E1: Sidecar JSON result cache

**Files:**

- Create: `internal/adapters/cache/sidecar.go`
- Create: `internal/adapters/cache/sidecar_test.go`

- [ ] **Step 1: Write the failing test**

```go
package cache

import (
    "path/filepath"
    "testing"
    "time"

    "github.com/stretchr/testify/require"

    "github.com/leotulipan/transcribe/internal/core/domain"
)

func TestSidecar_RoundTrip(t *testing.T) {
    dir := t.TempDir()
    input := filepath.Join(dir, "talk.mp3")
    require.NoError(t, writeEmpty(input))

    c := New()
    res := &domain.Result{
        Provider: domain.ProviderGroq,
        Model:    "whisper-large-v3",
        Language: "en",
        Text:     "hello world",
        Duration: 5 * time.Second,
        Words: []domain.Word{
            {Text: "hello", Start: 0, End: 500 * time.Millisecond},
            {Text: "world", Start: 600 * time.Millisecond, End: time.Second},
        },
        RawJSON: []byte(`{"raw":"yes"}`),
    }
    require.NoError(t, c.Save(input, res))

    out, hit, err := c.Lookup(input, domain.ProviderGroq)
    require.NoError(t, err)
    require.True(t, hit)
    require.Equal(t, "hello world", out.Text)
    require.Equal(t, "whisper-large-v3", out.Model)
    require.Equal(t, 5*time.Second, out.Duration)
    require.Len(t, out.Words, 2)
    require.Equal(t, 500*time.Millisecond, out.Words[0].End)
}

func TestSidecar_MissReturnsFalse(t *testing.T) {
    dir := t.TempDir()
    input := filepath.Join(dir, "absent.mp3")
    require.NoError(t, writeEmpty(input))

    _, hit, err := New().Lookup(input, domain.ProviderGroq)
    require.NoError(t, err)
    require.False(t, hit)
}

func TestSidecar_UnknownSchemaIsMiss(t *testing.T) {
    dir := t.TempDir()
    input := filepath.Join(dir, "x.mp3")
    require.NoError(t, writeEmpty(input))

    side := sidecarPath(input, domain.ProviderGroq)
    require.NoError(t, writeBytes(side, []byte(`{"schema_version":999}`)))

    _, hit, err := New().Lookup(input, domain.ProviderGroq)
    require.NoError(t, err)
    require.False(t, hit, "unknown schema version should be treated as miss")
}

func writeEmpty(path string) error  { return writeBytes(path, []byte("")) }
func writeBytes(path string, b []byte) error {
    return osWriteFile(path, b)
}
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
go test ./internal/adapters/cache/...
```

Expected: compile error (`New`, `sidecarPath`, `osWriteFile` undefined).

- [ ] **Step 3: Write `internal/adapters/cache/sidecar.go`**

```go
package cache

import (
    "encoding/json"
    "errors"
    "io/fs"
    "os"
    "path/filepath"
    "strings"
    "time"

    "github.com/leotulipan/transcribe/internal/core/domain"
)

const schemaVersion = 1

type envelope struct {
    SchemaVersion int                 `json:"schema_version"`
    Provider      domain.ProviderID   `json:"provider"`
    Model         string              `json:"model"`
    Language      string              `json:"language"`
    DurationMs    int64               `json:"duration_ms"`
    Text          string              `json:"text"`
    Confidence    float64             `json:"confidence"`
    Words         []wordJSON          `json:"words"`
    Segments      []segmentJSON       `json:"segments"`
    SourcePath    string              `json:"source_path"`
    Raw           json.RawMessage     `json:"raw,omitempty"`
}

type wordJSON struct {
    Text       string  `json:"text"`
    StartMs    int64   `json:"start_ms"`
    EndMs      int64   `json:"end_ms"`
    Confidence float64 `json:"confidence,omitempty"`
}

type segmentJSON struct {
    Text      string `json:"text"`
    StartMs   int64  `json:"start_ms"`
    EndMs     int64  `json:"end_ms"`
    SpeakerID string `json:"speaker_id,omitempty"`
}

type Sidecar struct{}

func New() *Sidecar { return &Sidecar{} }

// osWriteFile lives at package scope so tests can use it via the test helper.
var osWriteFile = func(path string, data []byte) error {
    if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
        return err
    }
    return os.WriteFile(path, data, 0o644)
}

// sidecarPath returns the on-disk path of the cache file for (input, provider).
func sidecarPath(inputPath string, p domain.ProviderID) string {
    base := strings.TrimSuffix(inputPath, filepath.Ext(inputPath))
    return base + ".transcribe." + string(p) + ".json"
}

func (s *Sidecar) Lookup(inputPath string, p domain.ProviderID) (*domain.Result, bool, error) {
    data, err := os.ReadFile(sidecarPath(inputPath, p))
    switch {
    case errors.Is(err, fs.ErrNotExist):
        return nil, false, nil
    case err != nil:
        return nil, false, err
    }
    var env envelope
    if err := json.Unmarshal(data, &env); err != nil {
        return nil, false, err
    }
    if env.SchemaVersion != schemaVersion {
        return nil, false, nil
    }
    res := &domain.Result{
        Provider:   env.Provider,
        Model:      env.Model,
        Language:   env.Language,
        Text:       env.Text,
        Confidence: env.Confidence,
        Duration:   time.Duration(env.DurationMs) * time.Millisecond,
        SourcePath: env.SourcePath,
        RawJSON:    []byte(env.Raw),
    }
    for _, w := range env.Words {
        res.Words = append(res.Words, domain.Word{
            Text:       w.Text,
            Start:      time.Duration(w.StartMs) * time.Millisecond,
            End:        time.Duration(w.EndMs) * time.Millisecond,
            Confidence: w.Confidence,
        })
    }
    for _, sg := range env.Segments {
        res.Segments = append(res.Segments, domain.Segment{
            Text:      sg.Text,
            Start:     time.Duration(sg.StartMs) * time.Millisecond,
            End:       time.Duration(sg.EndMs) * time.Millisecond,
            SpeakerID: sg.SpeakerID,
        })
    }
    return res, true, nil
}

func (s *Sidecar) Save(inputPath string, r *domain.Result) error {
    env := envelope{
        SchemaVersion: schemaVersion,
        Provider:      r.Provider,
        Model:         r.Model,
        Language:      r.Language,
        DurationMs:    r.Duration.Milliseconds(),
        Text:          r.Text,
        Confidence:    r.Confidence,
        SourcePath:    inputPath,
        Raw:           json.RawMessage(r.RawJSON),
    }
    for _, w := range r.Words {
        env.Words = append(env.Words, wordJSON{
            Text: w.Text, StartMs: w.Start.Milliseconds(), EndMs: w.End.Milliseconds(),
            Confidence: w.Confidence,
        })
    }
    for _, sg := range r.Segments {
        env.Segments = append(env.Segments, segmentJSON{
            Text: sg.Text, StartMs: sg.Start.Milliseconds(), EndMs: sg.End.Milliseconds(),
            SpeakerID: sg.SpeakerID,
        })
    }
    data, err := json.MarshalIndent(env, "", "  ")
    if err != nil {
        return err
    }
    return osWriteFile(sidecarPath(inputPath, r.Provider), data)
}
```

- [ ] **Step 4: Run the tests to confirm they pass**

```bash
go test ./internal/adapters/cache/...
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add internal/adapters/cache/
git commit -m "feat(cache): sidecar json result cache"
```

---

## Phase F — Format writers

### Task F1: Subtitle block grouping helper

**Files:**

- Create: `internal/adapters/format/grouping.go`
- Create: `internal/adapters/format/grouping_test.go`

- [ ] **Step 1: Write the failing test**

```go
package format

import (
    "testing"
    "time"

    "github.com/stretchr/testify/require"

    "github.com/leotulipan/transcribe/internal/core/domain"
)

func TestGroupWords_RespectsMaxWords(t *testing.T) {
    words := mkWords([]string{"one", "two", "three", "four", "five", "six", "seven", "eight"})
    blocks := groupWords(words, 7, 10*time.Second)
    require.GreaterOrEqual(t, len(blocks), 2)
    require.LessOrEqual(t, len(blocks[0].Words), 7)
}

func TestGroupWords_BreaksOnLongGap(t *testing.T) {
    words := []domain.Word{
        {Text: "hello", Start: 0, End: 500 * time.Millisecond},
        // big silent gap
        {Text: "world", Start: 5 * time.Second, End: 5500 * time.Millisecond},
    }
    blocks := groupWords(words, 7, 3*time.Second)
    require.Len(t, blocks, 2, "long gap should force a block break")
}

func mkWords(texts []string) []domain.Word {
    out := make([]domain.Word, len(texts))
    for i, t := range texts {
        start := time.Duration(i) * 400 * time.Millisecond
        out[i] = domain.Word{Text: t, Start: start, End: start + 300*time.Millisecond}
    }
    return out
}
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
go test ./internal/adapters/format/...
```

Expected: compile error (`groupWords` undefined).

- [ ] **Step 3: Write `internal/adapters/format/grouping.go`**

```go
package format

import (
    "time"

    "github.com/leotulipan/transcribe/internal/core/domain"
)

// block is one subtitle entry: a contiguous run of words sharing a start/end.
type block struct {
    Words []domain.Word
    Start time.Duration
    End   time.Duration
}

// groupWords folds a flat word list into subtitle blocks. A new block starts
// when the running block already has maxWords entries OR the gap between the
// previous word's End and the next word's Start exceeds maxGap.
func groupWords(words []domain.Word, maxWords int, maxGap time.Duration) []block {
    if len(words) == 0 {
        return nil
    }
    var (
        out []block
        cur block
    )
    cur.Words = []domain.Word{words[0]}
    cur.Start = words[0].Start
    cur.End = words[0].End

    for i := 1; i < len(words); i++ {
        w := words[i]
        gap := w.Start - cur.End
        if len(cur.Words) >= maxWords || gap > maxGap {
            out = append(out, cur)
            cur = block{Words: []domain.Word{w}, Start: w.Start, End: w.End}
            continue
        }
        cur.Words = append(cur.Words, w)
        cur.End = w.End
    }
    out = append(out, cur)
    return out
}

// formatTimecode renders an SRT-style timecode "HH:MM:SS,mmm".
func formatTimecode(d time.Duration) string {
    if d < 0 {
        d = 0
    }
    h := d / time.Hour
    d -= h * time.Hour
    m := d / time.Minute
    d -= m * time.Minute
    s := d / time.Second
    d -= s * time.Second
    ms := d / time.Millisecond
    return twoDigit(int(h)) + ":" + twoDigit(int(m)) + ":" + twoDigit(int(s)) + "," + threeDigit(int(ms))
}

func twoDigit(n int) string {
    if n < 10 {
        return "0" + itoa(n)
    }
    return itoa(n)
}

func threeDigit(n int) string {
    switch {
    case n < 10:
        return "00" + itoa(n)
    case n < 100:
        return "0" + itoa(n)
    default:
        return itoa(n)
    }
}

func itoa(n int) string {
    if n == 0 {
        return "0"
    }
    var buf [20]byte
    i := len(buf)
    for n > 0 {
        i--
        buf[i] = byte('0' + n%10)
        n /= 10
    }
    return string(buf[i:])
}
```

- [ ] **Step 4: Run the tests to confirm they pass**

```bash
go test ./internal/adapters/format/...
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add internal/adapters/format/grouping.go internal/adapters/format/grouping_test.go
git commit -m "feat(format): subtitle block grouping helper"
```

---

### Task F2: Text writer

**Files:**

- Create: `internal/adapters/format/text.go`
- Create: `internal/adapters/format/text_test.go`

- [ ] **Step 1: Write the failing test**

```go
package format

import (
    "os"
    "path/filepath"
    "testing"

    "github.com/stretchr/testify/require"

    "github.com/leotulipan/transcribe/internal/core/domain"
)

func TestText_Write(t *testing.T) {
    dir := t.TempDir()
    dst := filepath.Join(dir, "out.txt")
    w := NewText()
    require.Equal(t, domain.FormatText, w.Format())
    require.NoError(t, w.Write(&domain.Result{Text: "hello world\n"}, dst))
    got, err := os.ReadFile(dst)
    require.NoError(t, err)
    require.Equal(t, "hello world\n", string(got))
}
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
go test ./internal/adapters/format/... -run TestText
```

Expected: compile error (`NewText` undefined).

- [ ] **Step 3: Write `internal/adapters/format/text.go`**

```go
package format

import (
    "os"

    "github.com/leotulipan/transcribe/internal/core/domain"
)

type Text struct{}

func NewText() *Text { return &Text{} }

func (Text) Format() domain.OutputFormat { return domain.FormatText }

func (Text) Write(r *domain.Result, dst string) error {
    return os.WriteFile(dst, []byte(r.Text), 0o644)
}
```

- [ ] **Step 4: Run the tests**

```bash
go test ./internal/adapters/format/... -run TestText
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add internal/adapters/format/text.go internal/adapters/format/text_test.go
git commit -m "feat(format): text writer"
```

---

### Task F3: SRT writer

**Files:**

- Create: `internal/adapters/format/srt.go`
- Create: `internal/adapters/format/srt_test.go`
- Create: `internal/adapters/format/testdata/sample.srt.golden`

- [ ] **Step 1: Write the failing test**

```go
package format

import (
    "os"
    "path/filepath"
    "testing"
    "time"

    "github.com/stretchr/testify/require"

    "github.com/leotulipan/transcribe/internal/core/domain"
)

func TestSRT_Write_Golden(t *testing.T) {
    res := &domain.Result{
        Words: []domain.Word{
            {Text: "Hello",  Start: 1100 * time.Millisecond, End: 1500 * time.Millisecond},
            {Text: "world",  Start: 1600 * time.Millisecond, End: 2100 * time.Millisecond},
            {Text: "this",   Start: 2300 * time.Millisecond, End: 2600 * time.Millisecond},
            {Text: "is",     Start: 2700 * time.Millisecond, End: 2900 * time.Millisecond},
            {Text: "a",      Start: 3000 * time.Millisecond, End: 3100 * time.Millisecond},
            {Text: "test",   Start: 3200 * time.Millisecond, End: 3700 * time.Millisecond},
            // forced break by gap
            {Text: "second", Start: 8000 * time.Millisecond, End: 8500 * time.Millisecond},
            {Text: "block",  Start: 8600 * time.Millisecond, End: 9000 * time.Millisecond},
        },
    }
    dir := t.TempDir()
    dst := filepath.Join(dir, "out.srt")
    require.NoError(t, NewSRT().Write(res, dst))

    got, err := os.ReadFile(dst)
    require.NoError(t, err)

    golden, err := os.ReadFile("testdata/sample.srt.golden")
    require.NoError(t, err)
    require.Equal(t, string(golden), string(got))
}
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
go test ./internal/adapters/format/... -run TestSRT
```

Expected: compile error (`NewSRT` undefined).

- [ ] **Step 3: Write `internal/adapters/format/srt.go`**

```go
package format

import (
    "os"
    "strings"
    "time"

    "github.com/leotulipan/transcribe/internal/core/domain"
)

const (
    srtMaxWordsPerBlock = 7
    srtMaxGap           = 3 * time.Second
)

type SRT struct{}

func NewSRT() *SRT { return &SRT{} }

func (SRT) Format() domain.OutputFormat { return domain.FormatSRT }

func (SRT) Write(r *domain.Result, dst string) error {
    blocks := groupWords(r.Words, srtMaxWordsPerBlock, srtMaxGap)
    var b strings.Builder
    for i, blk := range blocks {
        b.WriteString(itoa(i + 1))
        b.WriteByte('\n')
        b.WriteString(formatTimecode(blk.Start))
        b.WriteString(" --> ")
        b.WriteString(formatTimecode(blk.End))
        b.WriteByte('\n')
        for j, w := range blk.Words {
            if j > 0 {
                b.WriteByte(' ')
            }
            b.WriteString(w.Text)
        }
        b.WriteString("\n\n")
    }
    return os.WriteFile(dst, []byte(b.String()), 0o644)
}
```

- [ ] **Step 4: Create the golden file**

Write `internal/adapters/format/testdata/sample.srt.golden`:

```
1
00:00:01,100 --> 00:00:03,700
Hello world this is a test

2
00:00:08,000 --> 00:00:09,000
second block

```

(Trailing blank line is intentional — every SRT block ends with `\n\n`.)

- [ ] **Step 5: Run the tests**

```bash
go test ./internal/adapters/format/... -run TestSRT
```

Expected: PASS. (If the diff is line-ending only, the golden file was saved as CRLF — open it in an editor that respects LF or use `git config core.autocrlf false` for this repo.)

- [ ] **Step 6: Commit**

```bash
git add internal/adapters/format/srt.go internal/adapters/format/srt_test.go internal/adapters/format/testdata/sample.srt.golden
git commit -m "feat(format): standard srt writer"
```

---

### Task F4: DaVinci SRT writer

**Files:**

- Create: `internal/adapters/format/davinci.go`
- Create: `internal/adapters/format/davinci_test.go`
- Create: `internal/adapters/format/testdata/sample.davinci.srt.golden`

The DaVinci writer reads two kinds of post-processing markers on words:
synthetic pause-marker words whose `Text == "(...)"` (inserted by the service's
`davinci.go` post-processor), and filler-word words tagged via the special
`Speaker.Label` convention — `services.davinci.Apply` sets word.Text to the
uppercase filler so we know to emit it on its own line.

For this task we focus on rendering the markers; the *insertion* lives in the
service layer (Phase K).

- [ ] **Step 1: Write the failing test**

```go
package format

import (
    "os"
    "path/filepath"
    "testing"
    "time"

    "github.com/stretchr/testify/require"

    "github.com/leotulipan/transcribe/internal/core/domain"
)

func TestDaVinci_Write_Golden(t *testing.T) {
    res := &domain.Result{
        Words: []domain.Word{
            {Text: "Wir",       Start: 1100 * time.Millisecond, End: 1300 * time.Millisecond},
            {Text: "testen",    Start: 1350 * time.Millisecond, End: 1700 * time.Millisecond},
            {Text: "ÄHM",       Start: 1750 * time.Millisecond, End: 1900 * time.Millisecond}, // filler (uppercase)
            {Text: "das",       Start: 2000 * time.Millisecond, End: 2200 * time.Millisecond},
            {Text: "Skript",    Start: 2300 * time.Millisecond, End: 2700 * time.Millisecond},
            {Text: "(...)",     Start: 2700 * time.Millisecond, End: 4500 * time.Millisecond}, // pause marker
            {Text: "Nochmal",   Start: 4500 * time.Millisecond, End: 5000 * time.Millisecond},
        },
    }
    dir := t.TempDir()
    dst := filepath.Join(dir, "out.davinci.srt")
    require.NoError(t, NewDaVinci().Write(res, dst))

    got, err := os.ReadFile(dst)
    require.NoError(t, err)

    golden, err := os.ReadFile("testdata/sample.davinci.srt.golden")
    require.NoError(t, err)
    require.Equal(t, string(golden), string(got))
}
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
go test ./internal/adapters/format/... -run TestDaVinci
```

Expected: compile error.

- [ ] **Step 3: Write `internal/adapters/format/davinci.go`**

```go
package format

import (
    "os"
    "strings"
    "time"
    "unicode"

    "github.com/leotulipan/transcribe/internal/core/domain"
)

const (
    davinciMaxWordsPerBlock = 7
    davinciMaxGap           = 3 * time.Second
)

type DaVinci struct{}

func NewDaVinci() *DaVinci { return &DaVinci{} }

func (DaVinci) Format() domain.OutputFormat { return domain.FormatDavinciSRT }

func (DaVinci) Write(r *domain.Result, dst string) error {
    var b strings.Builder
    index := 1

    // Walk words, emitting an SRT block per logical group. Filler-word and
    // pause-marker words break the current group and become their own block.
    var bucket []domain.Word
    flush := func() {
        if len(bucket) == 0 {
            return
        }
        writeBlock(&b, index, bucket)
        index++
        bucket = nil
    }
    for _, w := range r.Words {
        switch {
        case w.Text == "(...)":
            flush()
            writeBlock(&b, index, []domain.Word{w})
            index++
        case isAllUpper(w.Text):
            flush()
            writeBlock(&b, index, []domain.Word{w})
            index++
        default:
            // gap-induced break
            if len(bucket) > 0 {
                last := bucket[len(bucket)-1]
                if w.Start-last.End > davinciMaxGap || len(bucket) >= davinciMaxWordsPerBlock {
                    flush()
                }
            }
            bucket = append(bucket, w)
        }
    }
    flush()

    return os.WriteFile(dst, []byte(b.String()), 0o644)
}

func writeBlock(b *strings.Builder, idx int, words []domain.Word) {
    if len(words) == 0 {
        return
    }
    start := words[0].Start
    end := words[len(words)-1].End
    b.WriteString(itoa(idx))
    b.WriteByte('\n')
    b.WriteString(formatTimecode(start))
    b.WriteString(" --> ")
    b.WriteString(formatTimecode(end))
    b.WriteByte('\n')
    for i, w := range words {
        if i > 0 {
            b.WriteByte(' ')
        }
        b.WriteString(w.Text)
    }
    b.WriteString("\n\n")
}

func isAllUpper(s string) bool {
    if s == "" {
        return false
    }
    hasLetter := false
    for _, r := range s {
        if unicode.IsLetter(r) {
            hasLetter = true
            if !unicode.IsUpper(r) {
                return false
            }
        }
    }
    return hasLetter
}
```

- [ ] **Step 4: Create the golden file**

Write `internal/adapters/format/testdata/sample.davinci.srt.golden`:

```
1
00:00:01,100 --> 00:00:01,700
Wir testen

2
00:00:01,750 --> 00:00:01,900
ÄHM

3
00:00:02,000 --> 00:00:02,700
das Skript

4
00:00:02,700 --> 00:00:04,500
(...)

5
00:00:04,500 --> 00:00:05,000
Nochmal

```

- [ ] **Step 5: Run the tests**

```bash
go test ./internal/adapters/format/... -run TestDaVinci
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add internal/adapters/format/davinci.go internal/adapters/format/davinci_test.go internal/adapters/format/testdata/sample.davinci.srt.golden
git commit -m "feat(format): davinci srt writer"
```

---

## Phase G — FFmpeg adapter

> All ffmpeg tests skip when ffmpeg or ffprobe is not on PATH (`exec.LookPath`).
> CI without ffmpeg is fine — these only need to pass locally for the human
> running the plan on Windows 11.

### Task G1: Atomic *.partial → final rename helper

**Files:**

- Create: `internal/adapters/audio/atomic.go`
- Create: `internal/adapters/audio/atomic_test.go`

- [ ] **Step 1: Write the failing test**

```go
package audio

import (
    "os"
    "path/filepath"
    "testing"

    "github.com/stretchr/testify/require"
)

func TestPartialPath(t *testing.T) {
    require.Equal(t, "out.mp3.partial", partialPath("out.mp3"))
    require.Equal(t, filepath.Join("dir", "x.flac.partial"), partialPath(filepath.Join("dir", "x.flac")))
}

func TestPromote_RenamesAndReportsSize(t *testing.T) {
    dir := t.TempDir()
    final := filepath.Join(dir, "out.mp3")
    require.NoError(t, os.WriteFile(partialPath(final), []byte("12345"), 0o644))

    size, err := promote(final)
    require.NoError(t, err)
    require.Equal(t, int64(5), size)
    _, err = os.Stat(final)
    require.NoError(t, err)
    _, err = os.Stat(partialPath(final))
    require.ErrorIs(t, err, os.ErrNotExist)
}
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
go test ./internal/adapters/audio/... -run TestPartial -run TestPromote
```

Expected: compile error.

- [ ] **Step 3: Write `internal/adapters/audio/atomic.go`**

```go
package audio

import "os"

// partialPath returns the in-flight name used while ffmpeg is still writing.
func partialPath(finalPath string) string {
    return finalPath + ".partial"
}

// promote renames "<finalPath>.partial" to "<finalPath>" once ffmpeg exits 0.
// Returns the file size of the promoted file.
func promote(finalPath string) (int64, error) {
    if err := os.Rename(partialPath(finalPath), finalPath); err != nil {
        return 0, err
    }
    info, err := os.Stat(finalPath)
    if err != nil {
        return 0, err
    }
    return info.Size(), nil
}
```

- [ ] **Step 4: Run the tests**

```bash
go test ./internal/adapters/audio/... -run TestPartial -run TestPromote
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add internal/adapters/audio/atomic.go internal/adapters/audio/atomic_test.go
git commit -m "feat(audio): atomic partial→final rename helper"
```

---

### Task G2: Intermediate meta.json read/write

**Files:**

- Create: `internal/adapters/audio/meta.go`
- Create: `internal/adapters/audio/meta_test.go`

- [ ] **Step 1: Write the failing test**

```go
package audio

import (
    "path/filepath"
    "testing"

    "github.com/stretchr/testify/require"

    "github.com/leotulipan/transcribe/internal/core/domain"
)

func TestMeta_RoundTrip(t *testing.T) {
    dir := t.TempDir()
    intermediate := filepath.Join(dir, "x.m4a")

    in := MetaInfo{
        Operation:        "copy",
        SourcePath:       "C:/videos/in.mp4",
        SourceSize:       12345,
        SourceMTimeUnix:  1700000000,
        TargetCodec:      "aac",
        TargetContainer:  "m4a",
        MaxBytesBudget:   25 * 1024 * 1024,
        Provider:         domain.ProviderGroq,
        Model:            "whisper-large-v3",
    }
    require.NoError(t, WriteMeta(intermediate, in))

    out, err := ReadMeta(intermediate)
    require.NoError(t, err)
    require.Equal(t, in, out)
}
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
go test ./internal/adapters/audio/... -run TestMeta
```

Expected: compile error (`MetaInfo` undefined).

- [ ] **Step 3: Write `internal/adapters/audio/meta.go`**

```go
package audio

import (
    "encoding/json"
    "os"

    "github.com/leotulipan/transcribe/internal/core/domain"
)

const metaSchema = 1

// MetaInfo describes a cached intermediate audio file so the next run can
// decide whether it is reusable.
type MetaInfo struct {
    SchemaVersion   int               `json:"schema_version"`
    Operation       string            `json:"operation"` // "copy" | "transcode"
    SourcePath      string            `json:"source_path"`
    SourceSize      int64             `json:"source_size"`
    SourceMTimeUnix int64             `json:"source_mtime_unix"`
    TargetCodec     string            `json:"target_codec"`
    TargetContainer string            `json:"target_container"`
    MaxBytesBudget  int64             `json:"max_bytes_budget"`
    Provider        domain.ProviderID `json:"provider"`
    Model           string            `json:"model"`
}

func metaPath(intermediate string) string { return intermediate + ".meta.json" }

func WriteMeta(intermediate string, m MetaInfo) error {
    m.SchemaVersion = metaSchema
    data, err := json.MarshalIndent(m, "", "  ")
    if err != nil {
        return err
    }
    return os.WriteFile(metaPath(intermediate), data, 0o644)
}

func ReadMeta(intermediate string) (MetaInfo, error) {
    var m MetaInfo
    data, err := os.ReadFile(metaPath(intermediate))
    if err != nil {
        return m, err
    }
    if err := json.Unmarshal(data, &m); err != nil {
        return m, err
    }
    return m, nil
}
```

Note: the round-trip test compares the marshaled `MetaInfo` against the
input, so the `SchemaVersion: 1` round-trips. The test struct literal above
omits `SchemaVersion` — adjust by adding `SchemaVersion: metaSchema` to the
`in` literal so the equality holds, or change the test to compare individual
fields. Pick one:

> Adjust test: change `in := MetaInfo{` to `in := MetaInfo{SchemaVersion: metaSchema,` so the roundtrip equals.

- [ ] **Step 4: Apply the test adjustment from step 3 and rerun**

```bash
go test ./internal/adapters/audio/... -run TestMeta
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add internal/adapters/audio/meta.go internal/adapters/audio/meta_test.go
git commit -m "feat(audio): intermediate meta.json read/write"
```

---

### Task G3: FFmpeg constructor + Probe

**Files:**

- Create: `internal/adapters/audio/ffmpeg.go`
- Create: `internal/adapters/audio/probe.go`
- Create: `internal/adapters/audio/probe_test.go`
- Create: `testdata/short-sample.mp3` (place a ~1-2 second mp3 here; can be re-encoded from any source with `ffmpeg -i <any.wav> -t 1 -c:a libmp3lame testdata/short-sample.mp3`)

- [ ] **Step 1: Acquire a sample mp3**

Outside the test, produce a tiny mp3:

```powershell
ffmpeg -f lavfi -i "sine=frequency=440:duration=1" -c:a libmp3lame -b:a 64k testdata/short-sample.mp3
```

- [ ] **Step 2: Write the failing test**

```go
package audio

import (
    "os/exec"
    "testing"

    "github.com/stretchr/testify/require"
)

func skipIfNoFFmpeg(t *testing.T) {
    t.Helper()
    if _, err := exec.LookPath("ffmpeg"); err != nil {
        t.Skip("ffmpeg not on PATH")
    }
    if _, err := exec.LookPath("ffprobe"); err != nil {
        t.Skip("ffprobe not on PATH")
    }
}

func TestProbe_ReportsFormatAndCodec(t *testing.T) {
    skipIfNoFFmpeg(t)
    f, err := New("", "", nopLogger{})
    require.NoError(t, err)
    af, err := f.Probe("../../../testdata/short-sample.mp3")
    require.NoError(t, err)
    require.Equal(t, "mp3", af.Codec)
    require.Greater(t, af.SizeBytes, int64(0))
    require.Greater(t, int64(af.Duration), int64(0))
}

type nopLogger struct{}
func (nopLogger) Debug(string, ...any) {}
func (nopLogger) Info(string, ...any)  {}
func (nopLogger) Warn(string, ...any)  {}
func (nopLogger) Error(string, ...any) {}
```

- [ ] **Step 3: Run the test to confirm it fails**

```bash
go test ./internal/adapters/audio/... -run TestProbe
```

Expected: compile error (`New`, `Probe` undefined).

- [ ] **Step 4: Write `internal/adapters/audio/ffmpeg.go`**

```go
package audio

import (
    "errors"
    "os/exec"

    "github.com/leotulipan/transcribe/internal/core/domain"
    "github.com/leotulipan/transcribe/internal/ports"
)

// FFmpeg implements ports.AudioProcessor.
type FFmpeg struct {
    ffmpeg  string
    ffprobe string
    log     ports.Logger
}

func New(ffmpegPath, ffprobePath string, log ports.Logger) (*FFmpeg, error) {
    if ffmpegPath == "" {
        p, err := exec.LookPath("ffmpeg")
        if err != nil {
            return nil, domain.ErrFFmpegMissing
        }
        ffmpegPath = p
    }
    if ffprobePath == "" {
        p, err := exec.LookPath("ffprobe")
        if err != nil {
            return nil, domain.ErrFFmpegMissing
        }
        ffprobePath = p
    }
    return &FFmpeg{ffmpeg: ffmpegPath, ffprobe: ffprobePath, log: log}, nil
}

// compile-time check
var _ ports.AudioProcessor = (*FFmpeg)(nil)

// errInternal is a placeholder so unimplemented methods compile in early tasks.
var errInternal = errors.New("not implemented")
```

- [ ] **Step 5: Write `internal/adapters/audio/probe.go`**

```go
package audio

import (
    "context"
    "encoding/json"
    "fmt"
    "os"
    "os/exec"
    "path/filepath"
    "strconv"
    "strings"
    "time"

    "github.com/leotulipan/transcribe/internal/core/domain"
)

type ffprobeOutput struct {
    Format struct {
        Filename     string `json:"filename"`
        FormatName   string `json:"format_name"`
        Duration     string `json:"duration"`
        Size         string `json:"size"`
    } `json:"format"`
    Streams []struct {
        CodecType string `json:"codec_type"`
        CodecName string `json:"codec_name"`
    } `json:"streams"`
}

func (f *FFmpeg) Probe(path string) (domain.AudioFile, error) {
    out, err := exec.CommandContext(context.Background(),
        f.ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", path,
    ).Output()
    if err != nil {
        return domain.AudioFile{}, fmt.Errorf("ffprobe: %w", err)
    }
    var p ffprobeOutput
    if err := json.Unmarshal(out, &p); err != nil {
        return domain.AudioFile{}, err
    }
    af := domain.AudioFile{Path: path}
    af.Container = pickContainer(p.Format.FormatName, path)
    for _, s := range p.Streams {
        if s.CodecType == "audio" {
            af.Codec = s.CodecName
            break
        }
    }
    if sec, err := strconv.ParseFloat(p.Format.Duration, 64); err == nil {
        af.Duration = time.Duration(sec * float64(time.Second))
    }
    if n, err := strconv.ParseInt(p.Format.Size, 10, 64); err == nil {
        af.SizeBytes = n
    } else if info, err := os.Stat(path); err == nil {
        af.SizeBytes = info.Size()
    }
    return af, nil
}

// pickContainer picks the most useful container name. ffprobe's format_name is
// a comma-separated list (e.g. "mov,mp4,m4a,3gp"). Prefer the file extension
// when it appears in the list; otherwise take the first entry.
func pickContainer(formatName, path string) string {
    ext := strings.TrimPrefix(strings.ToLower(filepath.Ext(path)), ".")
    parts := strings.Split(formatName, ",")
    if ext != "" {
        for _, p := range parts {
            if p == ext {
                return ext
            }
        }
    }
    if len(parts) > 0 {
        return parts[0]
    }
    return ""
}
```

- [ ] **Step 6: Stub the remaining `AudioProcessor` methods so the build passes**

Append to `internal/adapters/audio/ffmpeg.go`:

```go
import "context"

func (f *FFmpeg) CopyAudio(ctx context.Context, in domain.AudioFile, workDir string) (domain.AudioFile, error) {
    return domain.AudioFile{}, errInternal
}
func (f *FFmpeg) ExtractAudio(ctx context.Context, videoPath, workDir string) (domain.AudioFile, error) {
    return domain.AudioFile{}, errInternal
}
func (f *FFmpeg) Transcode(ctx context.Context, in domain.AudioFile, t ports.TargetFormat, workDir string) (domain.AudioFile, error) {
    return domain.AudioFile{}, errInternal
}
func (f *FFmpeg) Chunk(ctx context.Context, in domain.AudioFile, maxBytes int64, workDir string) ([]domain.Chunk, error) {
    return nil, errInternal
}
func (f *FFmpeg) Cleanup(file domain.AudioFile) error { return errInternal }
```

(Tasks G4-G7 replace each stub with a real implementation.)

- [ ] **Step 7: Run the tests**

```bash
go test ./internal/adapters/audio/... -run TestProbe
```

Expected: PASS (or SKIP if no ffmpeg).

- [ ] **Step 8: Commit**

```bash
git add internal/adapters/audio/
git commit -m "feat(audio): ffmpeg constructor + probe"
```

---

### Task G4: CopyAudio

**Files:**

- Modify: `internal/adapters/audio/ffmpeg.go` (remove the `CopyAudio` stub)
- Create: `internal/adapters/audio/copy.go`
- Create: `internal/adapters/audio/copy_test.go`

- [ ] **Step 1: Write the failing test**

```go
package audio

import (
    "context"
    "os"
    "path/filepath"
    "testing"

    "github.com/stretchr/testify/require"
)

func TestCopyAudio_StreamCopiesIntoDerivedContainer(t *testing.T) {
    skipIfNoFFmpeg(t)
    f, err := New("", "", nopLogger{})
    require.NoError(t, err)

    src, err := f.Probe("../../../testdata/short-sample.mp3")
    require.NoError(t, err)

    workDir := t.TempDir()
    out, err := f.CopyAudio(context.Background(), src, workDir)
    require.NoError(t, err)
    require.True(t, out.IsTemp)
    require.True(t, out.Complete)
    require.Equal(t, "mp3", out.Container)
    require.Equal(t, "mp3", out.Codec)
    require.Greater(t, out.SizeBytes, int64(0))

    _, err = os.Stat(out.Path)
    require.NoError(t, err)
    require.Equal(t, filepath.Dir(out.Path), workDir)

    // partial file must not linger
    _, err = os.Stat(partialPath(out.Path))
    require.ErrorIs(t, err, os.ErrNotExist)
}
```

- [ ] **Step 2: Run to confirm it fails**

```bash
go test ./internal/adapters/audio/... -run TestCopyAudio
```

Expected: FAIL (`errInternal`).

- [ ] **Step 3: Remove the `CopyAudio` stub from ffmpeg.go and write `internal/adapters/audio/copy.go`**

```go
package audio

import (
    "context"
    "fmt"
    "os"
    "os/exec"
    "path/filepath"
    "strings"

    "github.com/leotulipan/transcribe/internal/core/domain"
)

// codecContainer maps an audio codec to the container we will wrap it in.
var codecContainer = map[string]string{
    "aac":       "m4a",
    "alac":      "m4a",
    "mp3":       "mp3",
    "opus":      "ogg",
    "vorbis":    "ogg",
    "flac":      "flac",
    "pcm_s16le": "wav",
    "pcm_s24le": "wav",
    "pcm_f32le": "wav",
}

// containerExt returns the file extension for a copy-target container.
func containerExt(container string) string {
    return "." + container
}

func (f *FFmpeg) CopyAudio(ctx context.Context, in domain.AudioFile, workDir string) (domain.AudioFile, error) {
    container, ok := codecContainer[in.Codec]
    if !ok {
        return domain.AudioFile{}, fmt.Errorf("copy-audio: codec %q has no known container", in.Codec)
    }
    if err := os.MkdirAll(workDir, 0o755); err != nil {
        return domain.AudioFile{}, err
    }
    base := strings.TrimSuffix(filepath.Base(in.Path), filepath.Ext(in.Path))
    final := filepath.Join(workDir, base+containerExt(container))
    partial := partialPath(final)

    cmd := exec.CommandContext(ctx, f.ffmpeg,
        "-y", "-i", in.Path, "-vn", "-c:a", "copy", partial,
    )
    if out, err := cmd.CombinedOutput(); err != nil {
        _ = os.Remove(partial)
        return domain.AudioFile{}, fmt.Errorf("ffmpeg copy: %w: %s", err, string(out))
    }
    size, err := promote(final)
    if err != nil {
        return domain.AudioFile{}, err
    }

    return domain.AudioFile{
        Path:      final,
        SizeBytes: size,
        Duration:  in.Duration,
        Container: container,
        Codec:     in.Codec,
        IsTemp:    true,
        Complete:  true,
    }, nil
}
```

Also delete the `CopyAudio` stub from `ffmpeg.go`.

- [ ] **Step 4: Run the tests**

```bash
go test ./internal/adapters/audio/...
```

Expected: PASS (or SKIP).

- [ ] **Step 5: Commit**

```bash
git add internal/adapters/audio/copy.go internal/adapters/audio/copy_test.go internal/adapters/audio/ffmpeg.go
git commit -m "feat(audio): copyaudio stream-copy"
```

---

### Task G5: ExtractAudio

**Files:**

- Modify: `internal/adapters/audio/ffmpeg.go` (remove the `ExtractAudio` stub)
- Create: `internal/adapters/audio/extract.go`
- Create: `internal/adapters/audio/extract_test.go`

- [ ] **Step 1: Write the failing test**

```go
package audio

import (
    "context"
    "testing"

    "github.com/stretchr/testify/require"
)

func TestExtractAudio_ProducesMonoPCMWav(t *testing.T) {
    skipIfNoFFmpeg(t)
    f, err := New("", "", nopLogger{})
    require.NoError(t, err)

    workDir := t.TempDir()
    out, err := f.ExtractAudio(context.Background(), "../../../testdata/short-sample.mp3", workDir)
    require.NoError(t, err)
    require.True(t, out.IsTemp)
    require.True(t, out.Complete)
    require.Equal(t, "wav", out.Container)
    require.Contains(t, out.Codec, "pcm")
}
```

- [ ] **Step 2: Write `internal/adapters/audio/extract.go`**

```go
package audio

import (
    "context"
    "fmt"
    "os"
    "os/exec"
    "path/filepath"
    "strings"

    "github.com/leotulipan/transcribe/internal/core/domain"
)

func (f *FFmpeg) ExtractAudio(ctx context.Context, videoPath, workDir string) (domain.AudioFile, error) {
    if err := os.MkdirAll(workDir, 0o755); err != nil {
        return domain.AudioFile{}, err
    }
    base := strings.TrimSuffix(filepath.Base(videoPath), filepath.Ext(videoPath))
    final := filepath.Join(workDir, base+".wav")
    partial := partialPath(final)

    cmd := exec.CommandContext(ctx, f.ffmpeg,
        "-y", "-i", videoPath, "-vn",
        "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
        partial,
    )
    if out, err := cmd.CombinedOutput(); err != nil {
        _ = os.Remove(partial)
        return domain.AudioFile{}, fmt.Errorf("ffmpeg extract: %w: %s", err, string(out))
    }
    size, err := promote(final)
    if err != nil {
        return domain.AudioFile{}, err
    }

    af, err := f.Probe(final)
    if err != nil {
        return domain.AudioFile{}, err
    }
    af.IsTemp = true
    af.Complete = true
    af.SizeBytes = size
    return af, nil
}
```

Remove the `ExtractAudio` stub from `ffmpeg.go`.

- [ ] **Step 3: Run the tests**

```bash
go test ./internal/adapters/audio/... -run TestExtract
```

Expected: PASS (or SKIP).

- [ ] **Step 4: Commit**

```bash
git add internal/adapters/audio/extract.go internal/adapters/audio/extract_test.go internal/adapters/audio/ffmpeg.go
git commit -m "feat(audio): extractaudio to mono pcm wav"
```

---

### Task G6: Transcode

**Files:**

- Modify: `internal/adapters/audio/ffmpeg.go` (remove the `Transcode` stub)
- Create: `internal/adapters/audio/transcode.go`
- Create: `internal/adapters/audio/transcode_test.go`

- [ ] **Step 1: Write the failing test**

```go
package audio

import (
    "context"
    "testing"

    "github.com/stretchr/testify/require"

    "github.com/leotulipan/transcribe/internal/ports"
)

func TestTranscode_MP3(t *testing.T) {
    skipIfNoFFmpeg(t)
    f, err := New("", "", nopLogger{})
    require.NoError(t, err)
    src, err := f.Probe("../../../testdata/short-sample.mp3")
    require.NoError(t, err)

    out, err := f.Transcode(context.Background(), src, ports.TargetFormat{
        Codec: "mp3", Bitrate: "64k",
    }, t.TempDir())
    require.NoError(t, err)
    require.True(t, out.Complete)
    require.Equal(t, "mp3", out.Codec)
    require.Equal(t, "mp3", out.Container)
    require.Greater(t, out.SizeBytes, int64(0))
}

func TestTranscode_FLAC(t *testing.T) {
    skipIfNoFFmpeg(t)
    f, err := New("", "", nopLogger{})
    require.NoError(t, err)
    src, err := f.Probe("../../../testdata/short-sample.mp3")
    require.NoError(t, err)

    out, err := f.Transcode(context.Background(), src, ports.TargetFormat{Codec: "flac"}, t.TempDir())
    require.NoError(t, err)
    require.Equal(t, "flac", out.Container)
    require.Equal(t, "flac", out.Codec)
}
```

- [ ] **Step 2: Write `internal/adapters/audio/transcode.go`**

```go
package audio

import (
    "context"
    "fmt"
    "os"
    "os/exec"
    "path/filepath"
    "strings"

    "github.com/leotulipan/transcribe/internal/core/domain"
    "github.com/leotulipan/transcribe/internal/ports"
)

// targetContainer returns the container/ext used to wrap a transcoded codec.
func targetContainer(codec string) (container, ext string, err error) {
    switch codec {
    case "mp3":
        return "mp3", ".mp3", nil
    case "flac":
        return "flac", ".flac", nil
    case "pcm_s16le":
        return "wav", ".wav", nil
    default:
        return "", "", fmt.Errorf("unsupported transcode codec %q", codec)
    }
}

// ffmpegCodecArgs returns the -c:a / -b:a / -ar / -ac args for a target.
func ffmpegCodecArgs(t ports.TargetFormat) []string {
    var args []string
    switch t.Codec {
    case "mp3":
        args = append(args, "-c:a", "libmp3lame")
    case "flac":
        args = append(args, "-c:a", "flac")
    case "pcm_s16le":
        args = append(args, "-c:a", "pcm_s16le")
    default:
        args = append(args, "-c:a", t.Codec)
    }
    if t.Bitrate != "" {
        args = append(args, "-b:a", t.Bitrate)
    }
    if t.SampleRate > 0 {
        args = append(args, "-ar", fmt.Sprintf("%d", t.SampleRate))
    }
    return args
}

func (f *FFmpeg) Transcode(ctx context.Context, in domain.AudioFile, t ports.TargetFormat, workDir string) (domain.AudioFile, error) {
    container, ext, err := targetContainer(t.Codec)
    if err != nil {
        return domain.AudioFile{}, err
    }
    if err := os.MkdirAll(workDir, 0o755); err != nil {
        return domain.AudioFile{}, err
    }
    base := strings.TrimSuffix(filepath.Base(in.Path), filepath.Ext(in.Path))
    final := filepath.Join(workDir, base+ext)
    partial := partialPath(final)

    args := []string{"-y", "-i", in.Path, "-vn"}
    args = append(args, ffmpegCodecArgs(t)...)
    args = append(args, partial)

    cmd := exec.CommandContext(ctx, f.ffmpeg, args...)
    if out, err := cmd.CombinedOutput(); err != nil {
        _ = os.Remove(partial)
        return domain.AudioFile{}, fmt.Errorf("ffmpeg transcode: %w: %s", err, string(out))
    }
    size, err := promote(final)
    if err != nil {
        return domain.AudioFile{}, err
    }

    return domain.AudioFile{
        Path:      final,
        SizeBytes: size,
        Duration:  in.Duration,
        Container: container,
        Codec:     t.Codec,
        IsTemp:    true,
        Complete:  true,
    }, nil
}
```

Remove the `Transcode` stub from `ffmpeg.go`.

- [ ] **Step 3: Run the tests**

```bash
go test ./internal/adapters/audio/... -run TestTranscode
```

Expected: PASS (or SKIP).

- [ ] **Step 4: Commit**

```bash
git add internal/adapters/audio/transcode.go internal/adapters/audio/transcode_test.go internal/adapters/audio/ffmpeg.go
git commit -m "feat(audio): transcode to mp3/flac/pcm"
```

---

### Task G7: Chunk + Cleanup

**Files:**

- Modify: `internal/adapters/audio/ffmpeg.go` (remove `Chunk` and `Cleanup` stubs)
- Create: `internal/adapters/audio/chunk.go`
- Create: `internal/adapters/audio/cleanup.go`
- Create: `internal/adapters/audio/chunk_test.go`

- [ ] **Step 1: Write the failing test**

```go
package audio

import (
    "context"
    "os"
    "path/filepath"
    "testing"

    "github.com/stretchr/testify/require"
)

func TestChunk_SplitsUnderMaxBytes(t *testing.T) {
    skipIfNoFFmpeg(t)
    f, err := New("", "", nopLogger{})
    require.NoError(t, err)
    src, err := f.Probe("../../../testdata/short-sample.mp3")
    require.NoError(t, err)

    workDir := t.TempDir()
    // Force at least 2 chunks by halving the budget.
    budget := src.SizeBytes/2 + 1
    chunks, err := f.Chunk(context.Background(), src, budget, workDir)
    require.NoError(t, err)
    require.GreaterOrEqual(t, len(chunks), 2)
    for _, c := range chunks {
        require.True(t, c.Complete)
        info, err := os.Stat(c.Path)
        require.NoError(t, err)
        require.Equal(t, filepath.Dir(c.Path), workDir)
        require.LessOrEqual(t, info.Size(), budget+1024*1024) // 1MB slack for header re-emission
    }
}

func TestCleanup_DeletesIntermediateAndMeta(t *testing.T) {
    skipIfNoFFmpeg(t)
    f, err := New("", "", nopLogger{})
    require.NoError(t, err)
    workDir := t.TempDir()
    src, err := f.Probe("../../../testdata/short-sample.mp3")
    require.NoError(t, err)
    out, err := f.CopyAudio(context.Background(), src, workDir)
    require.NoError(t, err)
    require.NoError(t, WriteMeta(out.Path, MetaInfo{Operation: "copy"}))

    require.NoError(t, f.Cleanup(out))
    _, err = os.Stat(out.Path)
    require.ErrorIs(t, err, os.ErrNotExist)
    _, err = os.Stat(metaPath(out.Path))
    require.ErrorIs(t, err, os.ErrNotExist)
}
```

- [ ] **Step 2: Write `internal/adapters/audio/chunk.go`**

```go
package audio

import (
    "context"
    "fmt"
    "os"
    "os/exec"
    "path/filepath"
    "strings"
    "time"

    "github.com/leotulipan/transcribe/internal/core/domain"
)

// Chunk slices `in` into chunks each ≤ maxBytes. Uses stream-copy so the chunk
// retains the source codec.
func (f *FFmpeg) Chunk(ctx context.Context, in domain.AudioFile, maxBytes int64, workDir string) ([]domain.Chunk, error) {
    if in.SizeBytes <= maxBytes {
        return []domain.Chunk{{Path: in.Path, StartOffset: 0, SizeBytes: in.SizeBytes, Complete: true}}, nil
    }
    if err := os.MkdirAll(workDir, 0o755); err != nil {
        return nil, err
    }

    // Bytes-per-second estimate from total file (margin: 90% of budget).
    if in.Duration <= 0 {
        return nil, fmt.Errorf("chunk: source duration unknown")
    }
    bps := float64(in.SizeBytes) / in.Duration.Seconds()
    chunkSec := (float64(maxBytes) * 0.9) / bps
    if chunkSec < 5 {
        chunkSec = 5
    }
    chunkDur := time.Duration(chunkSec * float64(time.Second))

    base := strings.TrimSuffix(filepath.Base(in.Path), filepath.Ext(in.Path))
    ext := filepath.Ext(in.Path)

    var chunks []domain.Chunk
    var offset time.Duration
    idx := 0
    for offset < in.Duration {
        idx++
        final := filepath.Join(workDir, fmt.Sprintf("%s-chunk%02d%s", base, idx, ext))
        partial := partialPath(final)

        cmd := exec.CommandContext(ctx, f.ffmpeg,
            "-y",
            "-ss", fmt.Sprintf("%.3f", offset.Seconds()),
            "-t", fmt.Sprintf("%.3f", chunkDur.Seconds()),
            "-i", in.Path,
            "-c", "copy",
            partial,
        )
        if out, err := cmd.CombinedOutput(); err != nil {
            _ = os.Remove(partial)
            return nil, fmt.Errorf("ffmpeg chunk: %w: %s", err, string(out))
        }
        size, err := promote(final)
        if err != nil {
            return nil, err
        }
        chunks = append(chunks, domain.Chunk{
            Path:        final,
            StartOffset: offset,
            SizeBytes:   size,
            Complete:    true,
        })
        offset += chunkDur
    }
    return chunks, nil
}
```

- [ ] **Step 3: Write `internal/adapters/audio/cleanup.go`**

```go
package audio

import (
    "errors"
    "io/fs"
    "os"

    "github.com/leotulipan/transcribe/internal/core/domain"
)

func (f *FFmpeg) Cleanup(file domain.AudioFile) error {
    if !file.IsTemp || file.Path == "" {
        return nil
    }
    var firstErr error
    for _, p := range []string{file.Path, metaPath(file.Path)} {
        if err := os.Remove(p); err != nil && !errors.Is(err, fs.ErrNotExist) && firstErr == nil {
            firstErr = err
        }
    }
    return firstErr
}
```

Remove the `Chunk` and `Cleanup` stubs from `ffmpeg.go`. The unused
`errInternal` placeholder can also be deleted now.

- [ ] **Step 4: Run the tests**

```bash
go test ./internal/adapters/audio/...
```

Expected: PASS (or SKIP).

- [ ] **Step 5: Commit**

```bash
git add internal/adapters/audio/chunk.go internal/adapters/audio/cleanup.go internal/adapters/audio/chunk_test.go internal/adapters/audio/ffmpeg.go
git commit -m "feat(audio): chunk + cleanup"
```

---

## Phase H — Retry helper

### Task H1: HTTP retry with jittered backoff

**Files:**

- Create: `internal/adapters/api/internal/retry/retry.go`
- Create: `internal/adapters/api/internal/retry/retry_test.go`

- [ ] **Step 1: Write the failing test**

```go
package retry

import (
    "context"
    "errors"
    "net"
    "testing"
    "time"

    "github.com/stretchr/testify/require"
)

type fakeNetErr struct{}
func (fakeNetErr) Error() string  { return "fake net timeout" }
func (fakeNetErr) Timeout() bool  { return true }
func (fakeNetErr) Temporary() bool { return true }
var _ net.Error = fakeNetErr{}

func TestDo_RetriesTransientAndSucceeds(t *testing.T) {
    var calls int
    err := Do(context.Background(), 3, 1*time.Millisecond, func() error {
        calls++
        if calls < 3 {
            return fakeNetErr{}
        }
        return nil
    })
    require.NoError(t, err)
    require.Equal(t, 3, calls)
}

func TestDo_DoesNotRetryPermanent(t *testing.T) {
    var calls int
    boom := errors.New("auth failure")
    err := Do(context.Background(), 5, 1*time.Millisecond, func() error {
        calls++
        return boom
    })
    require.ErrorIs(t, err, boom)
    require.Equal(t, 1, calls)
}

func TestDo_GivesUpAfterMaxAttempts(t *testing.T) {
    var calls int
    err := Do(context.Background(), 2, 1*time.Millisecond, func() error {
        calls++
        return fakeNetErr{}
    })
    require.Error(t, err)
    require.Equal(t, 2, calls)
}

func TestIsRetryable_HTTPStatus(t *testing.T) {
    require.True(t, IsRetryable(HTTPError{StatusCode: 500}))
    require.True(t, IsRetryable(HTTPError{StatusCode: 429}))
    require.True(t, IsRetryable(HTTPError{StatusCode: 503}))
    require.False(t, IsRetryable(HTTPError{StatusCode: 400}))
    require.False(t, IsRetryable(HTTPError{StatusCode: 401}))
}
```

- [ ] **Step 2: Run to confirm failure**

```bash
go test ./internal/adapters/api/internal/retry/...
```

Expected: compile error.

- [ ] **Step 3: Write `internal/adapters/api/internal/retry/retry.go`**

```go
package retry

import (
    "context"
    "errors"
    "fmt"
    "math/rand"
    "net"
    "time"
)

// HTTPError is a lightweight wrapper API adapters can return so the retry
// helper can classify by status code.
type HTTPError struct {
    StatusCode int
    Message    string
}

func (e HTTPError) Error() string {
    if e.Message != "" {
        return fmt.Sprintf("http %d: %s", e.StatusCode, e.Message)
    }
    return fmt.Sprintf("http %d", e.StatusCode)
}

// IsRetryable classifies common transient failures.
func IsRetryable(err error) bool {
    if err == nil {
        return false
    }
    var he HTTPError
    if errors.As(err, &he) {
        return he.StatusCode == 429 || (he.StatusCode >= 500 && he.StatusCode < 600)
    }
    var ne net.Error
    if errors.As(err, &ne) && ne.Timeout() {
        return true
    }
    if errors.Is(err, context.DeadlineExceeded) {
        return true
    }
    return false
}

// Do executes fn up to attempts times, returning the final error.
// Backoff: base * 2^(i-1) + jitter ∈ [0, base].
func Do(ctx context.Context, attempts int, base time.Duration, fn func() error) error {
    if attempts < 1 {
        attempts = 1
    }
    var err error
    for i := 1; i <= attempts; i++ {
        err = fn()
        if err == nil {
            return nil
        }
        if !IsRetryable(err) || i == attempts {
            return err
        }
        wait := base*time.Duration(1<<(i-1)) + time.Duration(rand.Int63n(int64(base)))
        select {
        case <-ctx.Done():
            return ctx.Err()
        case <-time.After(wait):
        }
    }
    return err
}
```

- [ ] **Step 4: Run the tests**

```bash
go test ./internal/adapters/api/internal/retry/...
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add internal/adapters/api/internal/retry/
git commit -m "feat(retry): jittered backoff with retryable classifier"
```

---

## Phase I — Groq provider

### Task I1: Models + capabilities

**Files:**

- Create: `internal/adapters/api/groq/models.go`
- Create: `internal/adapters/api/groq/models_test.go`

- [ ] **Step 1: Write the failing test**

```go
package groq

import (
    "testing"

    "github.com/stretchr/testify/require"

    "github.com/leotulipan/transcribe/internal/core/domain"
)

func TestDefaultModel(t *testing.T) {
    require.Equal(t, "whisper-large-v3", DefaultModel())
}

func TestCapabilities_DefaultModelHasWordTimestamps(t *testing.T) {
    c := Capabilities("whisper-large-v3")
    require.True(t, c.WordTimestamps)
    require.True(t, c.SegmentTimestamps)
    require.True(t, c.LanguageHint)
    require.Contains(t, c.AcceptedInputs, domain.AudioFormat{Codec: "mp3"})
    require.Contains(t, c.AcceptedInputs, domain.AudioFormat{Codec: "flac"})
}

func TestCapabilities_UnknownModelReturnsZero(t *testing.T) {
    c := Capabilities("totally-made-up")
    require.False(t, c.WordTimestamps)
    require.Empty(t, c.AcceptedInputs)
}
```

- [ ] **Step 2: Write `internal/adapters/api/groq/models.go`**

```go
package groq

import (
    "github.com/leotulipan/transcribe/internal/core/domain"
    "github.com/leotulipan/transcribe/internal/ports"
)

// Per https://console.groq.com/docs/speech-to-text
var modelCaps = map[string]ports.ModelCapabilities{
    "whisper-large-v3": {
        WordTimestamps:    true,
        SegmentTimestamps: true,
        LanguageHint:      true,
        AcceptedInputs: []domain.AudioFormat{
            {Codec: "mp3"}, {Codec: "mp4"}, {Codec: "aac"},
            {Codec: "flac"}, {Codec: "ogg"}, {Codec: "opus"},
            {Codec: "pcm_s16le"}, {Codec: "wav"},
        },
    },
    "whisper-large-v3-turbo": {
        WordTimestamps:    true,
        SegmentTimestamps: true,
        LanguageHint:      true,
        AcceptedInputs: []domain.AudioFormat{
            {Codec: "mp3"}, {Codec: "mp4"}, {Codec: "aac"},
            {Codec: "flac"}, {Codec: "ogg"}, {Codec: "opus"},
            {Codec: "pcm_s16le"}, {Codec: "wav"},
        },
    },
}

func Models() []string {
    out := make([]string, 0, len(modelCaps))
    for k := range modelCaps {
        out = append(out, k)
    }
    return out
}

func DefaultModel() string { return "whisper-large-v3" }

func Capabilities(model string) ports.ModelCapabilities {
    return modelCaps[model] // zero value if unknown — fail-safe
}
```

- [ ] **Step 3: Run the tests**

```bash
go test ./internal/adapters/api/groq/...
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add internal/adapters/api/groq/models.go internal/adapters/api/groq/models_test.go
git commit -m "feat(groq): model registry + capabilities"
```

---

### Task I2: Response parser

**Files:**

- Create: `internal/adapters/api/groq/parse.go`
- Create: `internal/adapters/api/groq/parse_test.go`
- Create: `testdata/groq_sample.json` (Groq's "verbose_json" response, capture once with curl or copy from python `test/sample_speech_*.json` if shape matches; below is a minimal hand-written fixture)

- [ ] **Step 1: Write `testdata/groq_sample.json`**

```json
{
  "text": "Hello world this is a test.",
  "language": "en",
  "duration": 5.0,
  "segments": [
    {"id": 0, "start": 0.0, "end": 2.5, "text": "Hello world", "avg_logprob": -0.1},
    {"id": 1, "start": 2.6, "end": 5.0, "text": "this is a test.", "avg_logprob": -0.2}
  ],
  "words": [
    {"word": "Hello",  "start": 0.0, "end": 0.5},
    {"word": "world",  "start": 0.6, "end": 1.2},
    {"word": "this",   "start": 2.6, "end": 2.9},
    {"word": "is",     "start": 3.0, "end": 3.2},
    {"word": "a",      "start": 3.3, "end": 3.4},
    {"word": "test.",  "start": 3.5, "end": 5.0}
  ]
}
```

- [ ] **Step 2: Write the failing test**

```go
package groq

import (
    "os"
    "testing"
    "time"

    "github.com/stretchr/testify/require"

    "github.com/leotulipan/transcribe/internal/core/domain"
)

func TestParse_Fixture(t *testing.T) {
    data, err := os.ReadFile("../../../../testdata/groq_sample.json")
    require.NoError(t, err)

    r, err := parse(data, "whisper-large-v3")
    require.NoError(t, err)

    require.Equal(t, "Hello world this is a test.", r.Text)
    require.Equal(t, "en", r.Language)
    require.Equal(t, 5*time.Second, r.Duration)
    require.Len(t, r.Words, 6)
    require.Equal(t, "Hello", r.Words[0].Text)
    require.Equal(t, 500*time.Millisecond, r.Words[0].End)
    require.Len(t, r.Segments, 2)
    require.Equal(t, domain.ProviderGroq, r.Provider)
    require.Equal(t, "whisper-large-v3", r.Model)
    require.NotEmpty(t, r.RawJSON)
}
```

- [ ] **Step 3: Run to confirm failure**

```bash
go test ./internal/adapters/api/groq/... -run TestParse
```

Expected: compile error (`parse` undefined).

- [ ] **Step 4: Write `internal/adapters/api/groq/parse.go`**

```go
package groq

import (
    "encoding/json"
    "time"

    "github.com/leotulipan/transcribe/internal/core/domain"
)

type response struct {
    Text     string    `json:"text"`
    Language string    `json:"language"`
    Duration float64   `json:"duration"`
    Segments []segment `json:"segments"`
    Words    []word    `json:"words"`
}

type segment struct {
    ID         int     `json:"id"`
    Start      float64 `json:"start"`
    End        float64 `json:"end"`
    Text       string  `json:"text"`
    AvgLogprob float64 `json:"avg_logprob"`
}

type word struct {
    Word  string  `json:"word"`
    Start float64 `json:"start"`
    End   float64 `json:"end"`
}

func parse(data []byte, model string) (*domain.Result, error) {
    var resp response
    if err := json.Unmarshal(data, &resp); err != nil {
        return nil, err
    }
    res := &domain.Result{
        Provider: domain.ProviderGroq,
        Model:    model,
        Language: resp.Language,
        Text:     resp.Text,
        Duration: time.Duration(resp.Duration * float64(time.Second)),
        RawJSON:  data,
    }
    for _, w := range resp.Words {
        res.Words = append(res.Words, domain.Word{
            Text:  w.Word,
            Start: time.Duration(w.Start * float64(time.Second)),
            End:   time.Duration(w.End * float64(time.Second)),
        })
    }
    for _, s := range resp.Segments {
        res.Segments = append(res.Segments, domain.Segment{
            Text:  s.Text,
            Start: time.Duration(s.Start * float64(time.Second)),
            End:   time.Duration(s.End * float64(time.Second)),
        })
    }
    return res, nil
}
```

- [ ] **Step 5: Run the tests**

```bash
go test ./internal/adapters/api/groq/...
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add internal/adapters/api/groq/parse.go internal/adapters/api/groq/parse_test.go testdata/groq_sample.json
git commit -m "feat(groq): response parser"
```

---

### Task I3: HTTP client implementing the Provider port

**Files:**

- Create: `internal/adapters/api/groq/client.go`
- Create: `internal/adapters/api/groq/client_test.go`

- [ ] **Step 1: Write the failing test**

```go
package groq

import (
    "context"
    "net/http"
    "net/http/httptest"
    "os"
    "path/filepath"
    "testing"

    "github.com/stretchr/testify/require"

    "github.com/leotulipan/transcribe/internal/core/domain"
    "github.com/leotulipan/transcribe/internal/ports"
)

func TestClient_TranscribePostsAndParses(t *testing.T) {
    fixture, err := os.ReadFile("../../../../testdata/groq_sample.json")
    require.NoError(t, err)

    srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        require.Equal(t, "POST", r.Method)
        require.Equal(t, "/openai/v1/audio/transcriptions", r.URL.Path)
        require.Equal(t, "Bearer test-key", r.Header.Get("Authorization"))
        require.NoError(t, r.ParseMultipartForm(32<<20))
        require.Equal(t, "whisper-large-v3", r.FormValue("model"))
        f, _, err := r.FormFile("file")
        require.NoError(t, err)
        defer f.Close()
        w.Header().Set("Content-Type", "application/json")
        _, _ = w.Write(fixture)
    }))
    defer srv.Close()

    // tiny on-disk file the client can stream
    audioPath := filepath.Join(t.TempDir(), "tiny.mp3")
    require.NoError(t, os.WriteFile(audioPath, []byte("\xff\xfb\x90\x00"), 0o644))

    c := NewWithEndpoint("test-key", srv.URL, http.DefaultClient)
    require.Equal(t, domain.ProviderGroq, c.ID())
    res, err := c.Transcribe(context.Background(),
        domain.AudioFile{Path: audioPath, Container: "mp3", Codec: "mp3", SizeBytes: 4},
        ports.ProviderOpts{Model: "whisper-large-v3"},
    )
    require.NoError(t, err)
    require.Equal(t, "Hello world this is a test.", res.Text)
}
```

- [ ] **Step 2: Run to confirm failure**

```bash
go test ./internal/adapters/api/groq/...
```

Expected: compile error.

- [ ] **Step 3: Write `internal/adapters/api/groq/client.go`**

```go
package groq

import (
    "bytes"
    "context"
    "fmt"
    "io"
    "mime/multipart"
    "net/http"
    "os"
    "path/filepath"
    "time"

    "github.com/leotulipan/transcribe/internal/adapters/api/internal/retry"
    "github.com/leotulipan/transcribe/internal/core/domain"
    "github.com/leotulipan/transcribe/internal/ports"
)

const (
    defaultEndpoint   = "https://api.groq.com"
    maxUploadBytes    = 25 * 1024 * 1024
    transcribePath    = "/openai/v1/audio/transcriptions"
    requestTimeout    = 5 * time.Minute
)

// Client implements ports.Provider against Groq's OpenAI-compatible endpoint.
type Client struct {
    apiKey   string
    endpoint string
    http     *http.Client
}

func New(apiKey string, h *http.Client) *Client {
    return NewWithEndpoint(apiKey, defaultEndpoint, h)
}

func NewWithEndpoint(apiKey, endpoint string, h *http.Client) *Client {
    if h == nil {
        h = &http.Client{Timeout: requestTimeout}
    }
    return &Client{apiKey: apiKey, endpoint: endpoint, http: h}
}

var _ ports.Provider = (*Client)(nil)

func (c *Client) ID() domain.ProviderID    { return domain.ProviderGroq }
func (c *Client) MaxUploadBytes() int64    { return maxUploadBytes }
func (c *Client) Models() []string         { return Models() }
func (c *Client) DefaultModel() string     { return DefaultModel() }
func (c *Client) Capabilities(m string) ports.ModelCapabilities {
    return Capabilities(m)
}

func (c *Client) Transcribe(ctx context.Context, audio domain.AudioFile, opts ports.ProviderOpts) (*domain.Result, error) {
    model := opts.Model
    if model == "" {
        model = DefaultModel()
    }
    var raw []byte
    err := retry.Do(ctx, 3, 5*time.Second, func() error {
        body, contentType, err := buildMultipart(audio.Path, model, opts.Language)
        if err != nil {
            return err
        }
        req, err := http.NewRequestWithContext(ctx, "POST", c.endpoint+transcribePath, body)
        if err != nil {
            return err
        }
        req.Header.Set("Authorization", "Bearer "+c.apiKey)
        req.Header.Set("Content-Type", contentType)

        resp, err := c.http.Do(req)
        if err != nil {
            return err
        }
        defer resp.Body.Close()
        data, err := io.ReadAll(resp.Body)
        if err != nil {
            return err
        }
        if resp.StatusCode/100 != 2 {
            return retry.HTTPError{StatusCode: resp.StatusCode, Message: string(data)}
        }
        raw = data
        return nil
    })
    if err != nil {
        return nil, &domain.ErrProvider{
            Provider:  domain.ProviderGroq,
            Retryable: retry.IsRetryable(err),
            Cause:     err,
        }
    }
    return parse(raw, model)
}

func buildMultipart(path, model, language string) (*bytes.Buffer, string, error) {
    f, err := os.Open(path)
    if err != nil {
        return nil, "", err
    }
    defer f.Close()
    buf := &bytes.Buffer{}
    mw := multipart.NewWriter(buf)
    if err := mw.WriteField("model", model); err != nil {
        return nil, "", err
    }
    if err := mw.WriteField("response_format", "verbose_json"); err != nil {
        return nil, "", err
    }
    if err := mw.WriteField("timestamp_granularities[]", "word"); err != nil {
        return nil, "", err
    }
    if err := mw.WriteField("timestamp_granularities[]", "segment"); err != nil {
        return nil, "", err
    }
    if language != "" {
        if err := mw.WriteField("language", language); err != nil {
            return nil, "", err
        }
    }
    fw, err := mw.CreateFormFile("file", filepath.Base(path))
    if err != nil {
        return nil, "", err
    }
    if _, err := io.Copy(fw, f); err != nil {
        return nil, "", err
    }
    if err := mw.Close(); err != nil {
        return nil, "", err
    }
    return buf, mw.FormDataContentType(), nil
}

// Unused import guard for fmt — used in earlier draft. Remove if linter complains.
var _ = fmt.Sprintf
```

(Strip the `var _ = fmt.Sprintf` line if your linter rejects it; the `fmt`
import can be removed entirely if not used.)

- [ ] **Step 4: Run the tests**

```bash
go test ./internal/adapters/api/groq/...
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add internal/adapters/api/groq/client.go internal/adapters/api/groq/client_test.go
git commit -m "feat(groq): http client + provider impl"
```

---

### Task I4: Integration test (gated by API key)

**Files:**

- Create: `internal/adapters/api/groq/integration_test.go`

- [ ] **Step 1: Write the integration test**

```go
//go:build integration

package groq

import (
    "context"
    "net/http"
    "os"
    "testing"

    "github.com/stretchr/testify/require"

    "github.com/leotulipan/transcribe/internal/core/domain"
    "github.com/leotulipan/transcribe/internal/ports"
)

func TestIntegration_Groq_Transcribe(t *testing.T) {
    key := os.Getenv("TRANSCRIBE_GROQ_KEY")
    if key == "" {
        t.Skip("TRANSCRIBE_GROQ_KEY not set")
    }
    c := New(key, http.DefaultClient)
    res, err := c.Transcribe(context.Background(),
        domain.AudioFile{Path: "../../../../testdata/short-sample.mp3", Container: "mp3", Codec: "mp3"},
        ports.ProviderOpts{Model: c.DefaultModel(), Language: "en"},
    )
    require.NoError(t, err)
    require.NotEmpty(t, res.Text)
}
```

- [ ] **Step 2: Run normally (should be skipped without tag)**

```bash
go test ./internal/adapters/api/groq/...
```

Expected: PASS (the file is excluded by build tag).

- [ ] **Step 3: Optionally run with the integration tag and a real key**

```powershell
$env:TRANSCRIBE_GROQ_KEY = "gsk_..."
go test -tags integration ./internal/adapters/api/groq/...
```

- [ ] **Step 4: Commit**

```bash
git add internal/adapters/api/groq/integration_test.go
git commit -m "test(groq): integration test gated by build tag"
```

---

## Phase J — Logging adapter

### Task J1: slog-backed logger

**Files:**

- Create: `internal/adapters/logging/slog.go`

- [ ] **Step 1: Write `internal/adapters/logging/slog.go`**

```go
package logging

import (
    "io"
    "log/slog"
    "os"

    "github.com/leotulipan/transcribe/internal/ports"
)

type slogLogger struct {
    inner *slog.Logger
}

func NewText(out io.Writer, level slog.Level) ports.Logger {
    if out == nil {
        out = os.Stderr
    }
    h := slog.NewTextHandler(out, &slog.HandlerOptions{Level: level})
    return &slogLogger{inner: slog.New(h)}
}

func NewJSON(out io.Writer, level slog.Level) ports.Logger {
    if out == nil {
        out = os.Stderr
    }
    h := slog.NewJSONHandler(out, &slog.HandlerOptions{Level: level})
    return &slogLogger{inner: slog.New(h)}
}

func NewDiscard() ports.Logger {
    return &slogLogger{inner: slog.New(slog.NewTextHandler(io.Discard, nil))}
}

func (l *slogLogger) Debug(msg string, kv ...any) { l.inner.Debug(msg, kv...) }
func (l *slogLogger) Info(msg string, kv ...any)  { l.inner.Info(msg, kv...) }
func (l *slogLogger) Warn(msg string, kv ...any)  { l.inner.Warn(msg, kv...) }
func (l *slogLogger) Error(msg string, kv ...any) { l.inner.Error(msg, kv...) }
```

- [ ] **Step 2: Build**

```bash
go build ./internal/adapters/logging/...
```

- [ ] **Step 3: Commit**

```bash
git add internal/adapters/logging/
git commit -m "feat(logging): slog adapter"
```

---

## Phase K — Core service

Phase K builds the service in seven tasks. Tasks K1-K4 are pure logic and
test with stdlib only. Task K5 introduces the in-memory fake adapters that
K6 and K7 use to exercise the full pipeline.

### Task K1: Service struct + provider registry

**Files:**

- Create: `internal/core/services/registry.go`
- Create: `internal/core/services/service.go`
- Create: `internal/core/services/service_test.go`

- [ ] **Step 1: Write the failing test**

```go
package services

import (
    "testing"

    "github.com/stretchr/testify/require"

    "github.com/leotulipan/transcribe/internal/core/domain"
    "github.com/leotulipan/transcribe/internal/ports"
)

type fakeProvider struct {
    id     domain.ProviderID
    models []string
}

func (f fakeProvider) ID() domain.ProviderID    { return f.id }
func (fakeProvider) MaxUploadBytes() int64      { return 25 << 20 }
func (f fakeProvider) Models() []string         { return f.models }
func (f fakeProvider) DefaultModel() string {
    if len(f.models) == 0 {
        return ""
    }
    return f.models[0]
}
func (fakeProvider) Capabilities(string) ports.ModelCapabilities { return ports.ModelCapabilities{} }
func (fakeProvider) Transcribe(_ any, _ any, _ any) (*domain.Result, error) {
    panic("not called in this test")
}

func TestService_ListProviders_ReturnsConfigured(t *testing.T) {
    svc := New(Deps{
        Providers: map[domain.ProviderID]ports.Provider{
            domain.ProviderGroq:   fakeProvider{id: domain.ProviderGroq, models: []string{"whisper-large-v3"}},
            domain.ProviderOpenAI: fakeProvider{id: domain.ProviderOpenAI, models: []string{"whisper-1"}},
        },
    })
    got := svc.ListProviders()
    require.ElementsMatch(t, []domain.ProviderID{domain.ProviderGroq, domain.ProviderOpenAI}, got)
}

func TestService_ListModels_ReturnsProviderModels(t *testing.T) {
    svc := New(Deps{
        Providers: map[domain.ProviderID]ports.Provider{
            domain.ProviderGroq: fakeProvider{id: domain.ProviderGroq, models: []string{"whisper-large-v3"}},
        },
    })
    got, err := svc.ListModels(domain.ProviderGroq)
    require.NoError(t, err)
    require.Equal(t, []string{"whisper-large-v3"}, got)
}

func TestService_ListModels_UnknownProvider(t *testing.T) {
    svc := New(Deps{Providers: map[domain.ProviderID]ports.Provider{}})
    _, err := svc.ListModels(domain.ProviderGroq)
    require.ErrorIs(t, err, domain.ErrProviderMissing)
}
```

(Note: `fakeProvider.Transcribe` keeps the right method count to satisfy
`ports.Provider`. The `_ any` signature is a stand-in — once we need to
actually invoke it, K6 introduces a fuller fake.)

- [ ] **Step 2: Write `internal/core/services/registry.go`**

```go
package services

import (
    "fmt"

    "github.com/leotulipan/transcribe/internal/core/domain"
    "github.com/leotulipan/transcribe/internal/ports"
)

func providerFor(deps Deps, id domain.ProviderID) (ports.Provider, error) {
    p, ok := deps.Providers[id]
    if !ok {
        return nil, fmt.Errorf("%w: %s", domain.ErrProviderMissing, id)
    }
    return p, nil
}
```

- [ ] **Step 3: Write `internal/core/services/service.go`**

```go
package services

import (
    "context"

    "github.com/leotulipan/transcribe/internal/core/domain"
    "github.com/leotulipan/transcribe/internal/ports"
)

// Deps bundles the output ports the service needs.
type Deps struct {
    Providers map[domain.ProviderID]ports.Provider
    Audio     ports.AudioProcessor
    Cache     ports.ResultCache
    Writers   map[domain.OutputFormat]ports.FormatWriter
    Log       ports.Logger
}

// Service implements ports.TranscribeService.
type Service struct {
    deps Deps
}

func New(deps Deps) *Service { return &Service{deps: deps} }

var _ ports.TranscribeService = (*Service)(nil)

func (s *Service) ListProviders() []domain.ProviderID {
    out := make([]domain.ProviderID, 0, len(s.deps.Providers))
    for k := range s.deps.Providers {
        out = append(out, k)
    }
    return out
}

func (s *Service) ListModels(id domain.ProviderID) ([]string, error) {
    p, err := providerFor(s.deps, id)
    if err != nil {
        return nil, err
    }
    return p.Models(), nil
}

// Submit is implemented in job.go (Task K2).
func (s *Service) Submit(ctx context.Context, req domain.Request) (ports.Job, error) {
    return s.submit(ctx, req)
}
```

- [ ] **Step 4: Run the test (expect compile error on `submit`)**

```bash
go test ./internal/core/services/... -run TestService_List
```

Expected: compile error — `s.submit` undefined. That's fine; K2 supplies it.

- [ ] **Step 5: Add a stub for `submit` so this task's tests run**

Append a temporary stub to `service.go`:

```go
func (s *Service) submit(_ context.Context, _ domain.Request) (ports.Job, error) {
    return nil, nil // implemented in K2
}
```

- [ ] **Step 6: Run the tests**

```bash
go test ./internal/core/services/... -run TestService_List
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add internal/core/services/
git commit -m "feat(services): service struct + provider registry"
```

---

### Task K2: Job lifecycle (Submit/Progress/Wait/Cancel)

**Files:**

- Create: `internal/core/services/job.go`
- Create: `internal/core/services/job_test.go`
- Modify: `internal/core/services/service.go` — remove the K1 stub for `submit`

- [ ] **Step 1: Write the failing test**

```go
package services

import (
    "context"
    "errors"
    "sync/atomic"
    "testing"
    "time"

    "github.com/stretchr/testify/require"

    "github.com/leotulipan/transcribe/internal/core/domain"
)

func TestJob_LifecycleSuccess(t *testing.T) {
    j := newJob(context.Background(), domain.Request{}, "id-1")
    var ran atomic.Bool
    go j.run(func(emit func(domain.ProgressEvent)) (*domain.Result, error) {
        emit(domain.ProgressEvent{Stage: domain.StageProbing})
        ran.Store(true)
        return &domain.Result{Text: "done"}, nil
    })

    var seen []domain.Stage
    for ev := range j.Progress() {
        seen = append(seen, ev.Stage)
    }
    res, err := j.Wait()
    require.NoError(t, err)
    require.Equal(t, "done", res.Text)
    require.True(t, ran.Load())
    require.Contains(t, seen, domain.StageProbing)
}

func TestJob_CancelStopsWaitingFn(t *testing.T) {
    j := newJob(context.Background(), domain.Request{}, "id-2")
    started := make(chan struct{})
    go j.run(func(emit func(domain.ProgressEvent)) (*domain.Result, error) {
        close(started)
        <-j.ctx.Done()
        return nil, j.ctx.Err()
    })
    <-started
    j.Cancel()
    _, err := j.Wait()
    require.ErrorIs(t, err, context.Canceled)
}

func TestJob_WaitIsRepeatable(t *testing.T) {
    j := newJob(context.Background(), domain.Request{}, "id-3")
    boom := errors.New("boom")
    go j.run(func(emit func(domain.ProgressEvent)) (*domain.Result, error) {
        return nil, boom
    })
    _, err1 := j.Wait()
    require.ErrorIs(t, err1, boom)
    _, err2 := j.Wait()
    require.ErrorIs(t, err2, boom)
}

func TestJob_ProgressClosedAfterDone(t *testing.T) {
    j := newJob(context.Background(), domain.Request{}, "id-4")
    go j.run(func(emit func(domain.ProgressEvent)) (*domain.Result, error) {
        return &domain.Result{}, nil
    })
    // Drain
    for range j.Progress() {
    }
    select {
    case <-j.Progress():
        // already closed; receive on closed chan returns immediately — OK
    case <-time.After(100 * time.Millisecond):
        t.Fatal("Progress channel should be closed after job ends")
    }
}
```

- [ ] **Step 2: Write `internal/core/services/job.go`**

```go
package services

import (
    "context"
    "sync"
    "time"

    "github.com/leotulipan/transcribe/internal/core/domain"
    "github.com/leotulipan/transcribe/internal/ports"
)

const progressBuffer = 32

type job struct {
    id       string
    req      domain.Request
    progress chan domain.ProgressEvent
    done     chan struct{}
    once     sync.Once
    cancelF  context.CancelFunc
    ctx      context.Context
    started  time.Time

    mu     sync.RWMutex
    result *domain.Result
    err    error
}

func newJob(parent context.Context, req domain.Request, id string) *job {
    ctx, cancel := context.WithCancel(parent)
    return &job{
        id:       id,
        req:      req,
        progress: make(chan domain.ProgressEvent, progressBuffer),
        done:     make(chan struct{}),
        ctx:      ctx,
        cancelF:  cancel,
        started:  time.Now(),
    }
}

var _ ports.Job = (*job)(nil)

func (j *job) ID() string                                { return j.id }
func (j *job) Progress() <-chan domain.ProgressEvent     { return j.progress }
func (j *job) Cancel() {
    j.once.Do(func() { j.cancelF() })
}

func (j *job) Wait() (*domain.Result, error) {
    <-j.done
    j.mu.RLock()
    defer j.mu.RUnlock()
    return j.result, j.err
}

// emit pushes a progress event. Drops the event if the buffer is full so a
// slow UI never blocks the pipeline.
func (j *job) emit(ev domain.ProgressEvent) {
    ev.Elapsed = time.Since(j.started)
    select {
    case j.progress <- ev:
    default:
    }
}

// run executes pipeline. The fn receives an emit function and returns the
// final (*Result, error). Always closes the progress channel and the done
// channel on exit.
func (j *job) run(fn func(emit func(domain.ProgressEvent)) (*domain.Result, error)) {
    defer close(j.progress)
    defer close(j.done)

    result, err := fn(j.emit)

    j.mu.Lock()
    j.result = result
    j.err = err
    j.mu.Unlock()
}
```

- [ ] **Step 3: Replace the K1 `submit` stub in `service.go`**

```go
func (s *Service) submit(parent context.Context, req domain.Request) (ports.Job, error) {
    j := newJob(parent, req, generateJobID())
    go j.run(func(emit func(domain.ProgressEvent)) (*domain.Result, error) {
        return pipelineRun(j.ctx, req, s.deps, emit)
    })
    return j, nil
}

func generateJobID() string {
    // crypto/rand-backed in real life; for v1 a timestamp is fine
    return time.Now().UTC().Format("20060102T150405.000000000")
}
```

Add `"context"` and `"time"` to the imports as needed.

Also add a temporary stub for `pipelineRun` at the bottom of `service.go`:

```go
// pipelineRun is implemented in pipeline.go (Task K6). This stub keeps the
// build green for K2 tests that don't drive the real pipeline.
func pipelineRun(_ context.Context, _ domain.Request, _ Deps, _ func(domain.ProgressEvent)) (*domain.Result, error) {
    return &domain.Result{}, nil
}
```

- [ ] **Step 4: Run the tests**

```bash
go test ./internal/core/services/... -run TestJob
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add internal/core/services/
git commit -m "feat(services): job lifecycle (submit/progress/wait/cancel)"
```

---

### Task K3: Transient error classifier

**Files:**

- Create: `internal/core/services/transient.go`
- Create: `internal/core/services/transient_test.go`

- [ ] **Step 1: Write the failing test**

```go
package services

import (
    "context"
    "errors"
    "testing"

    "github.com/stretchr/testify/require"

    "github.com/leotulipan/transcribe/internal/core/domain"
)

func TestTransient_Classification(t *testing.T) {
    cases := []struct {
        name string
        err  error
        want bool
    }{
        {"nil",                nil,                                                false},
        {"canceled",           context.Canceled,                                   false},
        {"deadline exceeded",  context.DeadlineExceeded,                           true},
        {"ErrCanceled",        domain.ErrCanceled,                                 false},
        {"ErrIncompatible",    domain.ErrIncompatible{Reason: "x"},                false},
        {"provider retryable", &domain.ErrProvider{Retryable: true, Cause: errors.New("503")}, true},
        {"provider permanent", &domain.ErrProvider{Retryable: false, Cause: errors.New("401")}, false},
        {"unknown",            errors.New("other"),                                false},
    }
    for _, c := range cases {
        t.Run(c.name, func(t *testing.T) {
            require.Equal(t, c.want, transient(c.err))
        })
    }
}
```

- [ ] **Step 2: Write `internal/core/services/transient.go`**

```go
package services

import (
    "context"
    "errors"

    "github.com/leotulipan/transcribe/internal/core/domain"
)

// transient reports whether an error is the kind that a future retry might
// succeed against — drives the keep-intermediate cleanup policy.
func transient(err error) bool {
    if err == nil {
        return false
    }
    if errors.Is(err, context.Canceled) || errors.Is(err, domain.ErrCanceled) {
        return false
    }
    if errors.Is(err, context.DeadlineExceeded) {
        return true
    }
    var pe *domain.ErrProvider
    if errors.As(err, &pe) {
        return pe.Retryable
    }
    return false
}
```

- [ ] **Step 3: Run the tests**

```bash
go test ./internal/core/services/... -run TestTransient
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add internal/core/services/transient.go internal/core/services/transient_test.go
git commit -m "feat(services): transient error classifier"
```

---

### Task K4: DaVinci post-processor

**Files:**

- Create: `internal/core/services/davinci.go`
- Create: `internal/core/services/davinci_test.go`

- [ ] **Step 1: Write the failing test**

```go
package services

import (
    "strings"
    "testing"
    "time"

    "github.com/stretchr/testify/require"

    "github.com/leotulipan/transcribe/internal/core/domain"
)

func TestDavinciApply_InsertsPauseAndUppercasesFillers(t *testing.T) {
    res := &domain.Result{
        Words: []domain.Word{
            {Text: "Wir",     Start: 1100 * time.Millisecond, End: 1300 * time.Millisecond},
            {Text: "ähm",     Start: 1350 * time.Millisecond, End: 1500 * time.Millisecond},
            {Text: "testen",  Start: 1600 * time.Millisecond, End: 2000 * time.Millisecond},
            // 2.0s gap (exceeds 1.5s threshold)
            {Text: "Nochmal", Start: 4000 * time.Millisecond, End: 4500 * time.Millisecond},
        },
    }
    applyDavinci(res, &domain.DaVinciOptions{
        SilentPortionThreshold: 1500 * time.Millisecond,
    })

    var texts []string
    for _, w := range res.Words {
        texts = append(texts, w.Text)
    }
    joined := strings.Join(texts, " ")
    require.Contains(t, joined, "ÄHM", "filler must be uppercased")
    require.Contains(t, joined, "(...)", "pause must be inserted")
    // pause start should match previous end, end should match next start
    var pauseIdx int
    for i, w := range res.Words {
        if w.Text == "(...)" {
            pauseIdx = i
        }
    }
    require.Equal(t, 2000*time.Millisecond, res.Words[pauseIdx].Start)
    require.Equal(t, 4000*time.Millisecond, res.Words[pauseIdx].End)
}

func TestDavinciApply_NoOpWhenOptsNil(t *testing.T) {
    res := &domain.Result{Words: []domain.Word{{Text: "hi"}}}
    applyDavinci(res, nil)
    require.Equal(t, "hi", res.Words[0].Text)
}
```

- [ ] **Step 2: Write `internal/core/services/davinci.go`**

```go
package services

import (
    "strings"
    "time"

    "github.com/leotulipan/transcribe/internal/core/domain"
)

// applyDavinci mutates result.Words in place: pause markers get inserted as
// synthetic words with text "(...)", and filler-word matches get uppercased
// so the DaVinci format writer renders them on their own line.
func applyDavinci(r *domain.Result, opts *domain.DaVinciOptions) {
    if opts == nil || len(r.Words) == 0 {
        return
    }
    fillers := opts.FillerWords
    if len(fillers) == 0 {
        fillers = domain.DefaultFillerWords
    }
    threshold := opts.SilentPortionThreshold
    if threshold <= 0 {
        threshold = 1500 * time.Millisecond
    }

    fillerSet := map[string]struct{}{}
    for _, f := range fillers {
        fillerSet[strings.ToLower(f)] = struct{}{}
    }

    var out []domain.Word
    prevEnd := time.Duration(-1)
    for i, w := range r.Words {
        if i > 0 && threshold > 0 {
            gap := w.Start - prevEnd
            if gap >= threshold {
                out = append(out, domain.Word{
                    Text:  "(...)",
                    Start: prevEnd,
                    End:   w.Start,
                })
            }
        }
        text := w.Text
        if _, ok := fillerSet[strings.ToLower(strings.TrimFunc(text, isPunct))]; ok {
            text = strings.ToUpper(text)
        }
        out = append(out, domain.Word{
            Text:       text,
            Start:      w.Start,
            End:        w.End,
            Confidence: w.Confidence,
        })
        prevEnd = w.End
    }
    r.Words = out
}

func isPunct(r rune) bool {
    switch r {
    case '.', ',', '!', '?', ';', ':':
        return true
    }
    return false
}
```

- [ ] **Step 3: Run the tests**

```bash
go test ./internal/core/services/... -run TestDavinci
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add internal/core/services/davinci.go internal/core/services/davinci_test.go
git commit -m "feat(services): davinci pause + filler post-processor"
```

---

### Task K5: Prepare decision tree + intermediate cache lookup

**Files:**

- Create: `internal/core/services/prepare.go`
- Create: `internal/core/services/prepare_test.go`
- Create: `internal/core/services/fakes_test.go` — in-memory fakes used here and in K6

- [ ] **Step 1: Write `internal/core/services/fakes_test.go`**

```go
package services

import (
    "context"
    "errors"
    "sync"

    "github.com/leotulipan/transcribe/internal/core/domain"
    "github.com/leotulipan/transcribe/internal/ports"
)

// fakeAudio records every call so tests can assert which path the prepare
// decision tree took.
type fakeAudio struct {
    mu          sync.Mutex
    probeOut    domain.AudioFile
    copyOut     domain.AudioFile
    extractOut  domain.AudioFile
    transcOut   domain.AudioFile
    chunkOut    []domain.Chunk
    cleanupErr  error

    probeCalls    int
    copyCalls     int
    extractCalls  int
    transcCalls   int
    chunkCalls    int
    cleanupCalls  int
}

func (f *fakeAudio) Probe(string) (domain.AudioFile, error) {
    f.mu.Lock(); defer f.mu.Unlock()
    f.probeCalls++
    return f.probeOut, nil
}
func (f *fakeAudio) CopyAudio(_ context.Context, in domain.AudioFile, _ string) (domain.AudioFile, error) {
    f.mu.Lock(); defer f.mu.Unlock()
    f.copyCalls++
    out := f.copyOut
    if out.Path == "" {
        out = in
        out.IsTemp = true
        out.Complete = true
    }
    return out, nil
}
func (f *fakeAudio) ExtractAudio(_ context.Context, _ string, _ string) (domain.AudioFile, error) {
    f.mu.Lock(); defer f.mu.Unlock()
    f.extractCalls++
    return f.extractOut, nil
}
func (f *fakeAudio) Transcode(_ context.Context, in domain.AudioFile, _ ports.TargetFormat, _ string) (domain.AudioFile, error) {
    f.mu.Lock(); defer f.mu.Unlock()
    f.transcCalls++
    out := f.transcOut
    if out.Path == "" {
        out = in
        out.IsTemp = true
        out.Complete = true
    }
    return out, nil
}
func (f *fakeAudio) Chunk(_ context.Context, in domain.AudioFile, _ int64, _ string) ([]domain.Chunk, error) {
    f.mu.Lock(); defer f.mu.Unlock()
    f.chunkCalls++
    if f.chunkOut != nil {
        return f.chunkOut, nil
    }
    return []domain.Chunk{{Path: in.Path, SizeBytes: in.SizeBytes, Complete: true}}, nil
}
func (f *fakeAudio) Cleanup(domain.AudioFile) error {
    f.mu.Lock(); defer f.mu.Unlock()
    f.cleanupCalls++
    return f.cleanupErr
}

var _ ports.AudioProcessor = (*fakeAudio)(nil)

// fakeProviderFull is a complete Provider implementation for K6 tests.
type fakeProviderFull struct {
    id            domain.ProviderID
    models        []string
    caps          map[string]ports.ModelCapabilities
    maxUpload     int64
    result        *domain.Result
    err           error
    transcribeFn  func(ctx context.Context, audio domain.AudioFile, opts ports.ProviderOpts) (*domain.Result, error)
}

func (f *fakeProviderFull) ID() domain.ProviderID    { return f.id }
func (f *fakeProviderFull) MaxUploadBytes() int64    { return f.maxUpload }
func (f *fakeProviderFull) Models() []string         { return f.models }
func (f *fakeProviderFull) DefaultModel() string {
    if len(f.models) > 0 {
        return f.models[0]
    }
    return ""
}
func (f *fakeProviderFull) Capabilities(m string) ports.ModelCapabilities {
    if c, ok := f.caps[m]; ok {
        return c
    }
    return ports.ModelCapabilities{}
}
func (f *fakeProviderFull) Transcribe(ctx context.Context, a domain.AudioFile, o ports.ProviderOpts) (*domain.Result, error) {
    if f.transcribeFn != nil {
        return f.transcribeFn(ctx, a, o)
    }
    if f.err != nil {
        return nil, f.err
    }
    if f.result == nil {
        return nil, errors.New("fake provider not configured")
    }
    return f.result, nil
}

var _ ports.Provider = (*fakeProviderFull)(nil)

// fakeCache is a map-backed ResultCache.
type fakeCache struct {
    mu    sync.Mutex
    store map[string]*domain.Result
    saves int
}

func newFakeCache() *fakeCache {
    return &fakeCache{store: map[string]*domain.Result{}}
}

func (c *fakeCache) Lookup(path string, _ domain.ProviderID) (*domain.Result, bool, error) {
    c.mu.Lock(); defer c.mu.Unlock()
    r, ok := c.store[path]
    return r, ok, nil
}
func (c *fakeCache) Save(path string, r *domain.Result) error {
    c.mu.Lock(); defer c.mu.Unlock()
    c.store[path] = r
    c.saves++
    return nil
}

var _ ports.ResultCache = (*fakeCache)(nil)

// recordingWriter captures writes for assertion.
type recordingWriter struct {
    format domain.OutputFormat
    paths  []string
}

func (w *recordingWriter) Format() domain.OutputFormat { return w.format }
func (w *recordingWriter) Write(_ *domain.Result, dst string) error {
    w.paths = append(w.paths, dst)
    return nil
}

var _ ports.FormatWriter = (*recordingWriter)(nil)
```

- [ ] **Step 2: Write the failing test**

```go
package services

import (
    "context"
    "testing"

    "github.com/stretchr/testify/require"

    "github.com/leotulipan/transcribe/internal/core/domain"
    "github.com/leotulipan/transcribe/internal/ports"
)

func TestPrepare_AsIsWhenAcceptedAndSmallEnough(t *testing.T) {
    audio := &fakeAudio{}
    src := domain.AudioFile{Path: "in.mp3", Codec: "mp3", Container: "mp3", SizeBytes: 100}
    caps := ports.ModelCapabilities{AcceptedInputs: []domain.AudioFormat{{Codec: "mp3"}}}

    out, err := prepare(context.Background(), audio, src, caps, 1024, "/tmp", ports.TargetFormat{Codec: "mp3"})
    require.NoError(t, err)
    require.Equal(t, "in.mp3", out.Path)
    require.False(t, out.IsTemp)
    require.Equal(t, 0, audio.copyCalls)
    require.Equal(t, 0, audio.transcCalls)
}

func TestPrepare_CopyWhenAcceptedButContainerCantBeStreamed(t *testing.T) {
    // mp4 container containing AAC — AAC is accepted but the mp4 box around
    // it must be stream-copied into m4a.
    audio := &fakeAudio{copyOut: domain.AudioFile{Path: "out.m4a", Codec: "aac", Container: "m4a", IsTemp: true, Complete: true, SizeBytes: 200}}
    src := domain.AudioFile{Path: "in.mp4", Codec: "aac", Container: "mp4", SizeBytes: 500}
    caps := ports.ModelCapabilities{AcceptedInputs: []domain.AudioFormat{{Codec: "aac"}}}

    out, err := prepare(context.Background(), audio, src, caps, 1024, "/tmp", ports.TargetFormat{Codec: "mp3"})
    require.NoError(t, err)
    require.Equal(t, "out.m4a", out.Path)
    require.Equal(t, 1, audio.copyCalls)
    require.Equal(t, 0, audio.transcCalls)
}

func TestPrepare_TranscodeWhenSourceTooLarge(t *testing.T) {
    audio := &fakeAudio{transcOut: domain.AudioFile{Path: "out.mp3", Codec: "mp3", Container: "mp3", IsTemp: true, Complete: true, SizeBytes: 500}}
    src := domain.AudioFile{Path: "in.wav", Codec: "pcm_s16le", Container: "wav", SizeBytes: 10000}
    caps := ports.ModelCapabilities{AcceptedInputs: []domain.AudioFormat{{Codec: "mp3"}}}

    out, err := prepare(context.Background(), audio, src, caps, 1024, "/tmp", ports.TargetFormat{Codec: "mp3"})
    require.NoError(t, err)
    require.Equal(t, "out.mp3", out.Path)
    require.Equal(t, 1, audio.transcCalls)
}
```

- [ ] **Step 3: Write `internal/core/services/prepare.go`**

```go
package services

import (
    "context"

    "github.com/leotulipan/transcribe/internal/core/domain"
    "github.com/leotulipan/transcribe/internal/ports"
)

// codecAccepted reports whether (container, codec) is in the model's accepted
// inputs list. A blank Container in the accepted entry means "any container".
func codecAccepted(caps ports.ModelCapabilities, af domain.AudioFile) (acceptedContainer bool) {
    for _, in := range caps.AcceptedInputs {
        if in.Codec != "" && in.Codec != af.Codec {
            continue
        }
        if in.Container == "" || in.Container == af.Container {
            return true
        }
    }
    return false
}

// codecOnlyAccepted reports whether the codec alone is accepted (any container).
func codecOnlyAccepted(caps ports.ModelCapabilities, codec string) bool {
    for _, in := range caps.AcceptedInputs {
        if in.Codec == codec {
            return true
        }
    }
    return false
}

// prepare implements the copy-first decision tree (spec §6.3 step 5).
func prepare(
    ctx context.Context,
    audio ports.AudioProcessor,
    src domain.AudioFile,
    caps ports.ModelCapabilities,
    maxBytes int64,
    workDir string,
    transcodeTarget ports.TargetFormat,
) (domain.AudioFile, error) {
    // 5a: as-is
    if codecAccepted(caps, src) && src.SizeBytes <= maxBytes {
        return src, nil
    }
    // 5b: stream copy when the codec is accepted but container isn't (or video wrapper)
    if codecOnlyAccepted(caps, src.Codec) {
        out, err := audio.CopyAudio(ctx, src, workDir)
        if err == nil && out.SizeBytes <= maxBytes {
            return out, nil
        }
        // fall through to transcode on either error or "still too big"
    }
    // 5c: transcode
    return audio.Transcode(ctx, src, transcodeTarget, workDir)
}
```

- [ ] **Step 4: Run the tests**

```bash
go test ./internal/core/services/... -run TestPrepare
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add internal/core/services/prepare.go internal/core/services/prepare_test.go internal/core/services/fakes_test.go
git commit -m "feat(services): copy-first prepare decision tree + test fakes"
```

---

### Task K6: Pipeline state machine

**Files:**

- Create: `internal/core/services/pipeline.go`
- Create: `internal/core/services/chunking.go`
- Create: `internal/core/services/pipeline_test.go`
- Modify: `internal/core/services/service.go` — delete the `pipelineRun` stub

- [ ] **Step 1: Write the failing test**

```go
package services

import (
    "context"
    "encoding/json"
    "os"
    "path/filepath"
    "testing"
    "time"

    "github.com/stretchr/testify/require"

    "github.com/leotulipan/transcribe/internal/core/domain"
    "github.com/leotulipan/transcribe/internal/ports"
)

func TestPipeline_HappyPath_TextOnly(t *testing.T) {
    dir := t.TempDir()
    inputPath := filepath.Join(dir, "talk.mp3")
    require.NoError(t, os.WriteFile(inputPath, []byte("\xff\xfb"), 0o644))

    audio := &fakeAudio{
        probeOut: domain.AudioFile{Path: inputPath, Codec: "mp3", Container: "mp3", SizeBytes: 100, Duration: time.Second},
    }
    prov := &fakeProviderFull{
        id:        domain.ProviderGroq,
        models:    []string{"whisper-large-v3"},
        maxUpload: 1024,
        caps: map[string]ports.ModelCapabilities{
            "whisper-large-v3": {
                WordTimestamps: true,
                AcceptedInputs: []domain.AudioFormat{{Codec: "mp3"}},
            },
        },
        result: &domain.Result{
            Text:    "hello world",
            Language: "en",
            Words: []domain.Word{
                {Text: "hello", Start: 0, End: 500 * time.Millisecond},
                {Text: "world", Start: 600 * time.Millisecond, End: 1100 * time.Millisecond},
            },
            RawJSON: json.RawMessage(`{"k":"v"}`),
        },
    }
    cache := newFakeCache()
    textWriter := &recordingWriter{format: domain.FormatText}

    deps := Deps{
        Providers: map[domain.ProviderID]ports.Provider{domain.ProviderGroq: prov},
        Audio:     audio,
        Cache:     cache,
        Writers:   map[domain.OutputFormat]ports.FormatWriter{domain.FormatText: textWriter},
    }

    svc := New(deps)
    job, err := svc.Submit(context.Background(), domain.Request{
        InputPath: inputPath,
        Provider:  domain.ProviderGroq,
        Model:     "whisper-large-v3",
        Formats:   []domain.OutputFormat{domain.FormatText},
    })
    require.NoError(t, err)

    var events []domain.Stage
    for ev := range job.Progress() {
        events = append(events, ev.Stage)
    }
    res, err := job.Wait()
    require.NoError(t, err)
    require.Equal(t, "hello world", res.Text)
    require.Equal(t, 1, cache.saves)
    require.Len(t, textWriter.paths, 1)
    require.Contains(t, events, domain.StageProbing)
    require.Contains(t, events, domain.StageTranscribing)
    require.Contains(t, events, domain.StageWriting)
    require.Contains(t, events, domain.StageDone)
}

func TestPipeline_RejectsIncompatibleFormat(t *testing.T) {
    dir := t.TempDir()
    inputPath := filepath.Join(dir, "talk.mp3")
    require.NoError(t, os.WriteFile(inputPath, []byte("\xff\xfb"), 0o644))

    prov := &fakeProviderFull{
        id:        domain.ProviderOpenAI,
        models:    []string{"gpt-4o-audio"},
        maxUpload: 1024,
        caps: map[string]ports.ModelCapabilities{
            "gpt-4o-audio": {WordTimestamps: false, AcceptedInputs: []domain.AudioFormat{{Codec: "mp3"}}},
        },
    }
    svc := New(Deps{
        Providers: map[domain.ProviderID]ports.Provider{domain.ProviderOpenAI: prov},
        Audio:     &fakeAudio{},
        Cache:     newFakeCache(),
        Writers:   map[domain.OutputFormat]ports.FormatWriter{},
    })

    job, err := svc.Submit(context.Background(), domain.Request{
        InputPath: inputPath, Provider: domain.ProviderOpenAI, Model: "gpt-4o-audio",
        Formats: []domain.OutputFormat{domain.FormatSRT},
    })
    require.NoError(t, err)
    _, err = job.Wait()
    var ei domain.ErrIncompatible
    require.ErrorAs(t, err, &ei)
    require.Equal(t, domain.FormatSRT, ei.Format)
}

func TestPipeline_ResultCacheHit_SkipsAudioAndProvider(t *testing.T) {
    dir := t.TempDir()
    inputPath := filepath.Join(dir, "talk.mp3")
    require.NoError(t, os.WriteFile(inputPath, []byte("\xff\xfb"), 0o644))

    cached := &domain.Result{Text: "cached text", Provider: domain.ProviderGroq}
    cache := newFakeCache()
    require.NoError(t, cache.Save(inputPath, cached))

    audio := &fakeAudio{} // probeOut not set — pipeline still calls Probe but won't call transcode
    audio.probeOut = domain.AudioFile{Path: inputPath, Codec: "mp3", Container: "mp3", SizeBytes: 100, Duration: time.Second}

    prov := &fakeProviderFull{id: domain.ProviderGroq, models: []string{"whisper-large-v3"},
        maxUpload: 1024,
        caps: map[string]ports.ModelCapabilities{
            "whisper-large-v3": {WordTimestamps: true, AcceptedInputs: []domain.AudioFormat{{Codec: "mp3"}}},
        },
        transcribeFn: func(context.Context, domain.AudioFile, ports.ProviderOpts) (*domain.Result, error) {
            t.Fatal("provider must not be called on cache hit")
            return nil, nil
        },
    }
    textWriter := &recordingWriter{format: domain.FormatText}

    svc := New(Deps{
        Providers: map[domain.ProviderID]ports.Provider{domain.ProviderGroq: prov},
        Audio:     audio,
        Cache:     cache,
        Writers:   map[domain.OutputFormat]ports.FormatWriter{domain.FormatText: textWriter},
    })
    job, err := svc.Submit(context.Background(), domain.Request{
        InputPath: inputPath, Provider: domain.ProviderGroq, Model: "whisper-large-v3",
        Formats: []domain.OutputFormat{domain.FormatText}, UseCache: true,
    })
    require.NoError(t, err)
    res, err := job.Wait()
    require.NoError(t, err)
    require.Equal(t, "cached text", res.Text)
    require.Len(t, textWriter.paths, 1)
}
```

- [ ] **Step 2: Write `internal/core/services/chunking.go`**

```go
package services

import (
    "encoding/json"

    "github.com/leotulipan/transcribe/internal/core/domain"
)

// mergeChunks concatenates per-chunk Results, offsetting timestamps by each
// chunk's StartOffset. RawJSON of multi-chunk results is emitted as a JSON
// array of the per-chunk raw payloads.
func mergeChunks(parts []*domain.Result, chunks []domain.Chunk) (*domain.Result, error) {
    if len(parts) == 0 {
        return nil, nil
    }
    if len(parts) == 1 {
        return parts[0], nil
    }
    base := *parts[0]
    base.Text = ""
    base.Words = nil
    base.Segments = nil

    var rawAll []json.RawMessage
    for i, p := range parts {
        off := chunks[i].StartOffset
        if i > 0 {
            base.Text += " "
        }
        base.Text += p.Text
        for _, w := range p.Words {
            base.Words = append(base.Words, domain.Word{
                Text: w.Text, Confidence: w.Confidence,
                Start: w.Start + off, End: w.End + off,
            })
        }
        for _, s := range p.Segments {
            base.Segments = append(base.Segments, domain.Segment{
                Text: s.Text, SpeakerID: s.SpeakerID,
                Start: s.Start + off, End: s.End + off,
            })
        }
        rawAll = append(rawAll, json.RawMessage(p.RawJSON))
    }
    bs, err := json.Marshal(rawAll)
    if err != nil {
        return nil, err
    }
    base.RawJSON = bs
    return &base, nil
}
```

- [ ] **Step 3: Write `internal/core/services/pipeline.go`**

```go
package services

import (
    "context"
    "errors"
    "fmt"
    "io/fs"
    "os"
    "path/filepath"
    "strings"

    "github.com/leotulipan/transcribe/internal/core/domain"
    "github.com/leotulipan/transcribe/internal/ports"
)

// pipelineRun executes the full transcription pipeline. Returns the final
// Result + error; emits progress events to `emit` as it walks the stages.
func pipelineRun(ctx context.Context, req domain.Request, deps Deps, emit func(domain.ProgressEvent)) (*domain.Result, error) {
    prov, err := providerFor(deps, req.Provider)
    if err != nil {
        return nil, err
    }
    model := req.Model
    if model == "" {
        model = prov.DefaultModel()
    }
    caps := prov.Capabilities(model)

    // Stage 0 — capability check
    if err := checkCapabilities(req, model, caps); err != nil {
        return nil, err
    }

    // Stage 1 — probe
    emit(domain.ProgressEvent{Stage: domain.StageProbing})
    src, err := deps.Audio.Probe(req.InputPath)
    if err != nil {
        return nil, fmt.Errorf("probe: %w", err)
    }

    // Stage 2 — result-cache lookup
    var cached *domain.Result
    if req.UseCache && deps.Cache != nil {
        r, hit, err := deps.Cache.Lookup(req.InputPath, req.Provider)
        if err != nil {
            if deps.Log != nil {
                deps.Log.Warn("result cache lookup failed", "err", err)
            }
        } else if hit {
            cached = r
        }
    }

    var result *domain.Result
    var tempFiles []domain.AudioFile

    // Deferred cleanup with policy from spec §6.3.1
    defer func() {
        for _, tf := range tempFiles {
            if !tf.IsTemp {
                continue
            }
            switch {
            case !tf.Complete:
                _ = deps.Audio.Cleanup(tf)
            case err == nil:
                _ = deps.Audio.Cleanup(tf)
            case transient(err):
                // keep
            default:
                // keep
            }
        }
    }()

    if cached == nil {
        // Stage 3 — working dir
        workDir, _ := resolveWorkDir(req.InputPath)

        // Stage 5 — prepare (skipping 4 intermediate-cache for v1 minimum;
        // see plan note below — full intermediate cache is a follow-up task)
        targetCodec := preferredCodecFor(req.Provider, caps)
        emit(domain.ProgressEvent{Stage: domain.StageCompressing})
        prepared, perr := prepare(ctx, deps.Audio, src, caps, prov.MaxUploadBytes(), workDir, ports.TargetFormat{Codec: targetCodec})
        if perr != nil {
            err = perr
            return nil, fmt.Errorf("prepare: %w", err)
        }
        if prepared.IsTemp {
            tempFiles = append(tempFiles, prepared)
        }

        // Stage 6 — chunk (single-chunk path is common)
        emit(domain.ProgressEvent{Stage: domain.StageChunking})
        chunks, cerr := deps.Audio.Chunk(ctx, prepared, prov.MaxUploadBytes(), workDir)
        if cerr != nil {
            err = cerr
            return nil, fmt.Errorf("chunk: %w", err)
        }

        // Stage 7 — transcribe each chunk
        emit(domain.ProgressEvent{Stage: domain.StageTranscribing})
        var parts []*domain.Result
        for i, c := range chunks {
            emit(domain.ProgressEvent{
                Stage:   domain.StageTranscribing,
                Percent: float64(i) / float64(len(chunks)),
                Message: fmt.Sprintf("chunk %d/%d", i+1, len(chunks)),
            })
            chunkAudio := domain.AudioFile{
                Path: c.Path, SizeBytes: c.SizeBytes, Codec: prepared.Codec, Container: prepared.Container,
                IsTemp: prepared.IsTemp, Complete: c.Complete,
            }
            r, terr := prov.Transcribe(ctx, chunkAudio, ports.ProviderOpts{Model: model, Language: req.Language})
            if terr != nil {
                err = terr
                return nil, terr
            }
            parts = append(parts, r)
        }
        merged, merr := mergeChunks(parts, chunks)
        if merr != nil {
            err = merr
            return nil, merr
        }
        merged.SourcePath = req.InputPath
        merged.Provider = req.Provider
        merged.Model = model
        result = merged
    } else {
        result = cached
    }

    // Stage 7 — davinci post-processing (only mutates words; deterministic)
    if hasFormat(req.Formats, domain.FormatDavinciSRT) {
        applyDavinci(result, req.DaVinciOpts)
    }

    // Stage 8 — cache write (only when we actually transcribed)
    if cached == nil && deps.Cache != nil {
        if werr := deps.Cache.Save(req.InputPath, result); werr != nil && deps.Log != nil {
            deps.Log.Warn("cache save failed", "err", werr)
        }
    }

    // Stage 9 — write outputs
    emit(domain.ProgressEvent{Stage: domain.StageWriting})
    for i, f := range req.Formats {
        w, ok := deps.Writers[f]
        if !ok {
            err = fmt.Errorf("no writer registered for format %q", f)
            return nil, err
        }
        dst := outputPath(req, f)
        if werr := w.Write(result, dst); werr != nil {
            err = werr
            return nil, werr
        }
        emit(domain.ProgressEvent{Stage: domain.StageWriting, Percent: float64(i+1) / float64(len(req.Formats))})
    }

    emit(domain.ProgressEvent{Stage: domain.StageDone})
    return result, nil
}

func checkCapabilities(req domain.Request, model string, caps ports.ModelCapabilities) error {
    for _, f := range req.Formats {
        if f.NeedsTimestamps() && !caps.WordTimestamps {
            return domain.ErrIncompatible{
                Provider: req.Provider, Model: model, Format: f,
                Reason: "model does not return word-level timestamps",
            }
        }
    }
    return nil
}

func hasFormat(formats []domain.OutputFormat, f domain.OutputFormat) bool {
    for _, x := range formats {
        if x == f {
            return true
        }
    }
    return false
}

// preferredCodecFor returns the preferred transcode target codec for a provider.
// Constrained to a codec that appears in caps.AcceptedInputs so the prepared
// file is actually accepted by the model.
func preferredCodecFor(p domain.ProviderID, caps ports.ModelCapabilities) string {
    pref := []string{}
    switch p {
    case domain.ProviderAssemblyAI, domain.ProviderElevenLabs:
        pref = []string{"flac", "mp3"}
    default:
        pref = []string{"mp3", "flac"}
    }
    for _, codec := range pref {
        if codecOnlyAccepted(caps, codec) {
            return codec
        }
    }
    return "mp3" // fallback; transcode will surface an error if rejected
}

// resolveWorkDir picks the per-job temp directory next to the source file,
// falling back to os.TempDir() if the source-adjacent path isn't writable.
func resolveWorkDir(inputPath string) (string, bool) {
    base := strings.TrimSuffix(filepath.Base(inputPath), filepath.Ext(inputPath))
    sideBySide := filepath.Join(filepath.Dir(inputPath), ".transcribe-tmp", base)
    if err := os.MkdirAll(sideBySide, 0o755); err == nil {
        return sideBySide, true
    } else if !errors.Is(err, fs.ErrPermission) {
        return sideBySide, true
    }
    fallback := filepath.Join(os.TempDir(), "transcribe-"+base)
    _ = os.MkdirAll(fallback, 0o755)
    return fallback, false
}

// outputPath returns the path for an output file. If req.OutputDir is set the
// output lands there with the source basename; otherwise it lands next to the
// source.
func outputPath(req domain.Request, f domain.OutputFormat) string {
    base := strings.TrimSuffix(filepath.Base(req.InputPath), filepath.Ext(req.InputPath))
    dir := req.OutputDir
    if dir == "" {
        dir = filepath.Dir(req.InputPath)
    }
    var ext string
    switch f {
    case domain.FormatText:
        ext = ".txt"
    case domain.FormatSRT:
        ext = ".srt"
    case domain.FormatDavinciSRT:
        ext = ".davinci.srt"
    default:
        ext = "." + string(f)
    }
    return filepath.Join(dir, base+ext)
}
```

- [ ] **Step 4: Remove the `pipelineRun` stub from `service.go`**

Delete the stub function added in K2. The real one in `pipeline.go` now satisfies the call.

- [ ] **Step 5: Run the tests**

```bash
go test ./internal/core/services/...
```

Expected: PASS (all three pipeline tests + earlier service/job/transient/davinci/prepare tests).

- [ ] **Step 6: Commit**

```bash
git add internal/core/services/
git commit -m "feat(services): pipeline state machine + chunk merging"
```

---

### Task K7: Intermediate cache lookup integration

> The spec describes step 4 (intermediate-cache lookup via meta.json sidecars).
> K6 omitted it to keep the first pipeline test small. K7 layers it in.

**Files:**

- Modify: `internal/core/services/pipeline.go`
- Create: `internal/core/services/intermediate_cache.go`
- Create: `internal/core/services/intermediate_cache_test.go`

- [ ] **Step 1: Write the failing test**

```go
package services

import (
    "os"
    "path/filepath"
    "testing"

    "github.com/stretchr/testify/require"

    "github.com/leotulipan/transcribe/internal/adapters/audio"
    "github.com/leotulipan/transcribe/internal/core/domain"
)

func TestLookupIntermediate_MatchesAndIgnoresMismatch(t *testing.T) {
    dir := t.TempDir()
    inter := filepath.Join(dir, "talk.m4a")
    require.NoError(t, os.WriteFile(inter, []byte("data"), 0o644))

    src := domain.AudioFile{Path: "/tmp/talk.mp4", Codec: "aac", Container: "mp4", SizeBytes: 1000}
    require.NoError(t, audio.WriteMeta(inter, audio.MetaInfo{
        Operation: "copy", SourcePath: src.Path, SourceSize: src.SizeBytes,
        SourceMTimeUnix: 0, TargetCodec: "aac", TargetContainer: "m4a",
        MaxBytesBudget: 25 << 20, Provider: domain.ProviderGroq, Model: "whisper-large-v3",
    }))

    // matching lookup
    got := lookupIntermediate(dir, src, 0, domain.ProviderGroq, "whisper-large-v3", 25<<20, "aac")
    require.NotNil(t, got)
    require.Equal(t, inter, got.Path)

    // mismatching codec → nil
    none := lookupIntermediate(dir, src, 0, domain.ProviderGroq, "whisper-large-v3", 25<<20, "mp3")
    require.Nil(t, none)
}
```

- [ ] **Step 2: Write `internal/core/services/intermediate_cache.go`**

```go
package services

import (
    "os"
    "path/filepath"

    "github.com/leotulipan/transcribe/internal/adapters/audio"
    "github.com/leotulipan/transcribe/internal/core/domain"
)

// lookupIntermediate scans workDir for a meta.json sidecar matching the
// (source, provider, model, budget, target-codec) tuple and returns the
// AudioFile pointing at it. Returns nil on no match.
func lookupIntermediate(workDir string, src domain.AudioFile, srcMTime int64, p domain.ProviderID, model string, budget int64, targetCodec string) *domain.AudioFile {
    entries, err := os.ReadDir(workDir)
    if err != nil {
        return nil
    }
    for _, e := range entries {
        if e.IsDir() {
            continue
        }
        name := e.Name()
        if filepath.Ext(name) == ".json" {
            continue
        }
        full := filepath.Join(workDir, name)
        m, err := audio.ReadMeta(full)
        if err != nil {
            continue
        }
        if m.Provider != p || m.Model != model || m.TargetCodec != targetCodec {
            continue
        }
        if m.SourceSize != src.SizeBytes || m.MaxBytesBudget != budget {
            continue
        }
        if m.SourceMTimeUnix != 0 && srcMTime != 0 && m.SourceMTimeUnix != srcMTime {
            continue
        }
        info, err := os.Stat(full)
        if err != nil {
            continue
        }
        return &domain.AudioFile{
            Path:      full,
            SizeBytes: info.Size(),
            Container: m.TargetContainer,
            Codec:     m.TargetCodec,
            IsTemp:    true,
            Complete:  true,
        }
    }
    return nil
}
```

- [ ] **Step 3: Wire `lookupIntermediate` into `pipeline.go` before the prepare call**

In `pipeline.go`, replace the `prepare` invocation block with:

```go
// Stage 4 — intermediate cache
var prepared domain.AudioFile
srcMTime := int64(0)
if info, statErr := os.Stat(req.InputPath); statErr == nil {
    srcMTime = info.ModTime().Unix()
}
targetCodec := preferredCodecFor(req.Provider, caps)
if hit := lookupIntermediate(workDir, src, srcMTime, req.Provider, model, prov.MaxUploadBytes(), targetCodec); hit != nil {
    prepared = *hit
} else {
    emit(domain.ProgressEvent{Stage: domain.StageCompressing})
    p, perr := prepare(ctx, deps.Audio, src, caps, prov.MaxUploadBytes(), workDir, ports.TargetFormat{Codec: targetCodec})
    if perr != nil {
        err = perr
        return nil, fmt.Errorf("prepare: %w", err)
    }
    prepared = p
    // Write meta sidecar so future runs find it
    if prepared.IsTemp {
        _ = audio.WriteMeta(prepared.Path, audio.MetaInfo{
            Operation:       "transcode-or-copy",
            SourcePath:      req.InputPath,
            SourceSize:      src.SizeBytes,
            SourceMTimeUnix: srcMTime,
            TargetCodec:     prepared.Codec,
            TargetContainer: prepared.Container,
            MaxBytesBudget:  prov.MaxUploadBytes(),
            Provider:        req.Provider,
            Model:           model,
        })
    }
}
if prepared.IsTemp {
    tempFiles = append(tempFiles, prepared)
}
```

Add `"github.com/leotulipan/transcribe/internal/adapters/audio"` to the imports.

- [ ] **Step 4: Run the tests**

```bash
go test ./internal/core/services/...
```

Expected: PASS (the new lookup test plus all earlier tests).

- [ ] **Step 5: Commit**

```bash
git add internal/core/services/intermediate_cache.go internal/core/services/intermediate_cache_test.go internal/core/services/pipeline.go
git commit -m "feat(services): intermediate-cache lookup before prepare"
```

---

## Phase L — CLI delivery

### Task L1: Composition root (wire.go)

**Files:**

- Create: `internal/delivery/wire.go`

- [ ] **Step 1: Write `internal/delivery/wire.go`**

```go
package delivery

import (
    "net/http"

    "github.com/leotulipan/transcribe/internal/adapters/api/groq"
    "github.com/leotulipan/transcribe/internal/adapters/audio"
    "github.com/leotulipan/transcribe/internal/adapters/cache"
    "github.com/leotulipan/transcribe/internal/adapters/format"
    "github.com/leotulipan/transcribe/internal/core/domain"
    "github.com/leotulipan/transcribe/internal/core/services"
    "github.com/leotulipan/transcribe/internal/ports"
)

// BuildService wires the adapters into a TranscribeService.
func BuildService(cfg ports.Config, log ports.Logger) (ports.TranscribeService, error) {
    ffmpeg, err := audio.New(cfg.FFmpegPath, "", log)
    if err != nil {
        return nil, err
    }

    httpClient := &http.Client{} // groq.New sets its own timeout
    providers := map[domain.ProviderID]ports.Provider{}
    if key := cfg.APIKeys[domain.ProviderGroq]; key != "" {
        providers[domain.ProviderGroq] = groq.New(key, httpClient)
    }

    writers := map[domain.OutputFormat]ports.FormatWriter{
        domain.FormatText:       format.NewText(),
        domain.FormatSRT:        format.NewSRT(),
        domain.FormatDavinciSRT: format.NewDaVinci(),
    }

    return services.New(services.Deps{
        Providers: providers,
        Audio:     ffmpeg,
        Cache:     cache.New(),
        Writers:   writers,
        Log:       log,
    }), nil
}
```

- [ ] **Step 2: Build**

```bash
go build ./internal/delivery/...
```

- [ ] **Step 3: Commit**

```bash
git add internal/delivery/wire.go
git commit -m "feat(delivery): wire composition root"
```

---

### Task L2: Cobra root + transcribe command (CLI mode only)

**Files:**

- Create: `internal/delivery/cli/root.go`
- Create: `internal/delivery/cli/transcribe.go`
- Create: `cmd/transcribe/main.go`

- [ ] **Step 1: Add the cobra dependency**

```bash
go get github.com/spf13/cobra@latest
```

- [ ] **Step 2: Write `internal/delivery/cli/root.go`**

```go
package cli

import (
    "github.com/spf13/cobra"

    "github.com/leotulipan/transcribe/internal/ports"
)

type Deps struct {
    Service ports.TranscribeService
    Config  ports.Config
    Logger  ports.Logger
    Version string
}

func NewRoot(d Deps) *cobra.Command {
    root := &cobra.Command{
        Use:           "transcribe",
        Short:         "Transcribe audio and video files via multiple AI providers",
        Version:       d.Version,
        SilenceUsage:  true,
        SilenceErrors: true,
    }
    root.AddCommand(newTranscribeCmd(d))
    root.AddCommand(newProvidersCmd(d))
    root.AddCommand(newSetupCmd(d))
    return root
}
```

- [ ] **Step 3: Write `internal/delivery/cli/transcribe.go`**

```go
package cli

import (
    "context"
    "fmt"
    "os"
    "strings"

    "github.com/spf13/cobra"

    "github.com/leotulipan/transcribe/internal/core/domain"
)

type transcribeFlags struct {
    api      string
    model    string
    language string
    outputs  []string
    outDir   string
    cache    bool
    davinci  bool
    silentMs int
    jsonMode bool
    progress bool
}

func newTranscribeCmd(d Deps) *cobra.Command {
    f := &transcribeFlags{}
    cmd := &cobra.Command{
        Use:   "transcribe [flags] <file> [<file>...]",
        Short: "Transcribe one or more files",
        Args:  cobra.MinimumNArgs(1),
        RunE: func(c *cobra.Command, args []string) error {
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
    cmd.Flags().BoolVar(&f.jsonMode, "json", false, "agent-callable JSON output, no TUI escalation")
    cmd.Flags().BoolVar(&f.progress, "progress", false, "with --json, emit JSONL progress events")
    return cmd
}

func runTranscribe(ctx context.Context, d Deps, f *transcribeFlags, files []string) error {
    formats, err := parseFormats(f.outputs, f.davinci)
    if err != nil {
        return err
    }
    provider := domain.ProviderID(f.api)
    for _, file := range files {
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
            req.DaVinciOpts = &domain.DaVinciOptions{
                SilentPortionThreshold: parseSilentMs(f.silentMs),
            }
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
```

- [ ] **Step 4: Write a tiny `parseSilentMs` helper**

Append to `internal/delivery/cli/transcribe.go`:

```go
func parseSilentMs(ms int) time.Duration { return time.Duration(ms) * time.Millisecond }
```

Add `"time"` to the import block.

- [ ] **Step 5: Write `cmd/transcribe/main.go`**

```go
package main

import (
    "context"
    "fmt"
    "log/slog"
    "os"

    "github.com/leotulipan/transcribe/internal/adapters/config"
    "github.com/leotulipan/transcribe/internal/adapters/logging"
    "github.com/leotulipan/transcribe/internal/delivery"
    "github.com/leotulipan/transcribe/internal/delivery/cli"
)

var version = "dev"

func main() {
    cfg, err := config.New().Load()
    if err != nil {
        fmt.Fprintln(os.Stderr, "config:", err)
        os.Exit(3)
    }
    log := logging.NewText(os.Stderr, slog.LevelInfo)

    svc, err := delivery.BuildService(cfg, log)
    if err != nil {
        fmt.Fprintln(os.Stderr, err)
        os.Exit(3)
    }

    root := cli.NewRoot(cli.Deps{Service: svc, Config: cfg, Logger: log, Version: version})
    ctx, cancel := installSignals(context.Background())
    defer cancel()
    if err := root.ExecuteContext(ctx); err != nil {
        fmt.Fprintln(os.Stderr, err)
        os.Exit(cli.ExitCodeFor(err))
    }
}
```

- [ ] **Step 6: Add temporary stubs so the build is green**

Create `internal/delivery/cli/providers.go`:

```go
package cli

import "github.com/spf13/cobra"

func newProvidersCmd(d Deps) *cobra.Command {
    return &cobra.Command{Use: "providers", Short: "list configured providers (impl in L6)"}
}
```

Create `internal/delivery/cli/setup.go`:

```go
package cli

import "github.com/spf13/cobra"

func newSetupCmd(d Deps) *cobra.Command {
    return &cobra.Command{Use: "setup", Short: "non-interactive setup (impl in L6)"}
}
```

Create `internal/delivery/cli/exit.go`:

```go
package cli

// ExitCodeFor maps an error to a POSIX exit code per the spec table.
// Implementation in L5 covers all cases; this stub keeps the build green.
func ExitCodeFor(err error) int {
    if err == nil {
        return 0
    }
    return 1
}
```

Create `internal/delivery/cli/json_render.go`:

```go
package cli

import (
    "io"

    "github.com/leotulipan/transcribe/internal/core/domain"
)

// renderJSON is implemented in L4. This stub keeps the build green.
func renderJSON(_ io.Writer, _ interface {
    Progress() <-chan domain.ProgressEvent
    Wait() (*domain.Result, error)
}, _ bool) error {
    return nil
}
```

Create `cmd/transcribe/signals.go`:

```go
package main

import "context"

// installSignals is fleshed out in L5. Stub keeps the build green.
func installSignals(parent context.Context) (context.Context, context.CancelFunc) {
    return context.WithCancel(parent)
}
```

- [ ] **Step 7: Build**

```bash
go build ./...
```

Expected: success.

- [ ] **Step 8: Commit**

```bash
git add cmd internal/delivery/
git commit -m "feat(cli): cobra root + transcribe command (stubbed sub-features)"
```

---

### Task L3: Exit code mapper

**Files:**

- Modify: `internal/delivery/cli/exit.go`
- Create: `internal/delivery/cli/exit_test.go`

- [ ] **Step 1: Write the failing test**

```go
package cli

import (
    "errors"
    "testing"

    "github.com/stretchr/testify/require"

    "github.com/leotulipan/transcribe/internal/core/domain"
)

func TestExitCodeFor(t *testing.T) {
    require.Equal(t, 0, ExitCodeFor(nil))
    require.Equal(t, 3, ExitCodeFor(domain.ErrFFmpegMissing))
    require.Equal(t, 3, ExitCodeFor(domain.ErrConfigMissing))
    require.Equal(t, 3, ExitCodeFor(domain.ErrProviderMissing))
    require.Equal(t, 2, ExitCodeFor(domain.ErrIncompatible{}))
    require.Equal(t, 4, ExitCodeFor(&domain.ErrProvider{}))
    require.Equal(t, 130, ExitCodeFor(domain.ErrCanceled))
    require.Equal(t, 1, ExitCodeFor(errors.New("unexpected")))
}
```

- [ ] **Step 2: Replace the stub in `internal/delivery/cli/exit.go`**

```go
package cli

import (
    "errors"

    "github.com/leotulipan/transcribe/internal/core/domain"
)

const (
    exitOK             = 0
    exitInternal       = 1
    exitUsage          = 2
    exitConfig         = 3
    exitProvider       = 4
    exitAudio          = 5
    exitCanceled       = 130
)

func ExitCodeFor(err error) int {
    if err == nil {
        return exitOK
    }
    if errors.Is(err, domain.ErrFFmpegMissing) {
        return exitConfig
    }
    if errors.Is(err, domain.ErrConfigMissing) {
        return exitConfig
    }
    if errors.Is(err, domain.ErrProviderMissing) {
        return exitConfig
    }
    if errors.Is(err, domain.ErrCanceled) {
        return exitCanceled
    }
    var ei domain.ErrIncompatible
    if errors.As(err, &ei) {
        return exitUsage
    }
    var ep *domain.ErrProvider
    if errors.As(err, &ep) {
        return exitProvider
    }
    return exitInternal
}
```

- [ ] **Step 3: Run the tests**

```bash
go test ./internal/delivery/cli/...
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add internal/delivery/cli/exit.go internal/delivery/cli/exit_test.go
git commit -m "feat(cli): exit code mapper"
```

---

### Task L4: JSON renderer (event + result + error)

**Files:**

- Modify: `internal/delivery/cli/json_render.go`
- Create: `internal/delivery/cli/json_render_test.go`

- [ ] **Step 1: Write the failing test**

```go
package cli

import (
    "bytes"
    "encoding/json"
    "errors"
    "testing"
    "time"

    "github.com/stretchr/testify/require"

    "github.com/leotulipan/transcribe/internal/core/domain"
)

type fakeJob struct {
    events []domain.ProgressEvent
    res    *domain.Result
    err    error
}

func (f *fakeJob) Progress() <-chan domain.ProgressEvent {
    ch := make(chan domain.ProgressEvent, len(f.events))
    for _, ev := range f.events {
        ch <- ev
    }
    close(ch)
    return ch
}
func (f *fakeJob) Wait() (*domain.Result, error) { return f.res, f.err }

func TestRenderJSON_FinalOnly_Success(t *testing.T) {
    job := &fakeJob{res: &domain.Result{Text: "hi", Provider: domain.ProviderGroq}}
    var buf bytes.Buffer
    require.NoError(t, renderJSON(&buf, job, false))
    var got map[string]any
    require.NoError(t, json.Unmarshal(buf.Bytes(), &got))
    require.Equal(t, float64(1), got["schema_version"])
    require.Equal(t, "ok", got["status"])
    require.NotNil(t, got["result"])
}

func TestRenderJSON_FinalOnly_Error(t *testing.T) {
    job := &fakeJob{err: domain.ErrIncompatible{Provider: domain.ProviderGroq, Format: domain.FormatSRT, Reason: "no timestamps"}}
    var buf bytes.Buffer
    require.Error(t, renderJSON(&buf, job, false))
    var got map[string]any
    require.NoError(t, json.Unmarshal(buf.Bytes(), &got))
    require.Equal(t, "error", got["status"])
}

func TestRenderJSON_Stream_EmitsProgressLines(t *testing.T) {
    job := &fakeJob{
        events: []domain.ProgressEvent{
            {Stage: domain.StageProbing, Elapsed: 10 * time.Millisecond},
            {Stage: domain.StageTranscribing, Percent: 0.5, Elapsed: 50 * time.Millisecond},
        },
        res: &domain.Result{Text: "ok"},
    }
    var buf bytes.Buffer
    require.NoError(t, renderJSON(&buf, job, true))
    lines := bytes.Split(bytes.TrimRight(buf.Bytes(), "\n"), []byte("\n"))
    require.GreaterOrEqual(t, len(lines), 3) // 2 progress + 1 result
    var last map[string]any
    require.NoError(t, json.Unmarshal(lines[len(lines)-1], &last))
    require.Equal(t, "result", last["type"])
}

func TestRenderJSON_Stream_EmitsErrorLine(t *testing.T) {
    boom := errors.New("nope")
    job := &fakeJob{err: boom}
    var buf bytes.Buffer
    require.Error(t, renderJSON(&buf, job, true))
    require.Contains(t, buf.String(), `"type":"error"`)
}
```

- [ ] **Step 2: Replace the stub in `internal/delivery/cli/json_render.go`**

```go
package cli

import (
    "encoding/json"
    "errors"
    "io"
    "time"

    "github.com/leotulipan/transcribe/internal/core/domain"
)

const jsonSchemaVersion = 1

type jobLike interface {
    Progress() <-chan domain.ProgressEvent
    Wait() (*domain.Result, error)
}

type resultJSON struct {
    Provider domain.ProviderID `json:"provider"`
    Model    string            `json:"model"`
    Language string            `json:"language"`
    Text     string            `json:"text"`
    DurationMs int64           `json:"duration_ms"`
}

type errorJSON struct {
    Code    string                 `json:"code"`
    Message string                 `json:"message"`
    Details map[string]any         `json:"details,omitempty"`
}

func renderJSON(w io.Writer, job jobLike, stream bool) error {
    enc := json.NewEncoder(w)
    if stream {
        for ev := range job.Progress() {
            _ = enc.Encode(map[string]any{
                "type":    "progress",
                "stage":   ev.Stage.String(),
                "percent": ev.Percent,
                "elapsed_ms": ev.Elapsed.Milliseconds(),
                "message": ev.Message,
            })
        }
        res, err := job.Wait()
        if err != nil {
            _ = enc.Encode(map[string]any{
                "type":  "error",
                "error": errorPayload(err),
            })
            return err
        }
        _ = enc.Encode(map[string]any{
            "type":   "result",
            "result": toResultJSON(res),
        })
        return nil
    }

    // Drain progress without emitting (final-only mode)
    for range job.Progress() {
    }
    res, err := job.Wait()
    if err != nil {
        _ = enc.Encode(map[string]any{
            "schema_version": jsonSchemaVersion,
            "status":         "error",
            "error":          errorPayload(err),
        })
        return err
    }
    _ = enc.Encode(map[string]any{
        "schema_version": jsonSchemaVersion,
        "status":         "ok",
        "result":         toResultJSON(res),
    })
    return nil
}

func toResultJSON(r *domain.Result) resultJSON {
    return resultJSON{
        Provider:   r.Provider,
        Model:      r.Model,
        Language:   r.Language,
        Text:       r.Text,
        DurationMs: int64(r.Duration / time.Millisecond),
    }
}

func errorPayload(err error) errorJSON {
    if err == nil {
        return errorJSON{}
    }
    var ei domain.ErrIncompatible
    if errors.As(err, &ei) {
        return errorJSON{
            Code: "incompatible", Message: err.Error(),
            Details: map[string]any{
                "provider": ei.Provider, "model": ei.Model, "format": ei.Format, "reason": ei.Reason,
            },
        }
    }
    var ep *domain.ErrProvider
    if errors.As(err, &ep) {
        return errorJSON{
            Code: "provider", Message: err.Error(),
            Details: map[string]any{"provider": ep.Provider, "status": ep.StatusCode, "retryable": ep.Retryable},
        }
    }
    if errors.Is(err, domain.ErrFFmpegMissing) {
        return errorJSON{Code: "ffmpeg_missing", Message: err.Error()}
    }
    if errors.Is(err, domain.ErrProviderMissing) {
        return errorJSON{Code: "provider_missing", Message: err.Error()}
    }
    return errorJSON{Code: "internal", Message: err.Error()}
}
```

- [ ] **Step 3: Run the tests**

```bash
go test ./internal/delivery/cli/...
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add internal/delivery/cli/json_render.go internal/delivery/cli/json_render_test.go
git commit -m "feat(cli): json renderer for agent mode"
```

---

### Task L5: Signal handling

**Files:**

- Replace: `cmd/transcribe/signals.go`

- [ ] **Step 1: Replace the stub in `cmd/transcribe/signals.go`**

```go
package main

import (
    "context"
    "os"
    "os/signal"
    "syscall"
)

// installSignals returns a context that cancels on SIGINT or SIGTERM.
func installSignals(parent context.Context) (context.Context, context.CancelFunc) {
    ctx, cancel := context.WithCancel(parent)
    ch := make(chan os.Signal, 1)
    signal.Notify(ch, syscall.SIGINT, syscall.SIGTERM)
    go func() {
        <-ch
        cancel()
        signal.Stop(ch)
    }()
    return ctx, cancel
}
```

- [ ] **Step 2: Build**

```bash
go build ./cmd/transcribe/...
```

- [ ] **Step 3: Commit**

```bash
git add cmd/transcribe/signals.go
git commit -m "feat(cli): signal handling cancels jobs"
```

---

### Task L6: `providers` and `setup` subcommands

**Files:**

- Replace: `internal/delivery/cli/providers.go`
- Replace: `internal/delivery/cli/setup.go`

- [ ] **Step 1: Replace `providers.go`**

```go
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
```

- [ ] **Step 2: Replace `setup.go`**

```go
package cli

import (
    "fmt"

    "github.com/spf13/cobra"

    "github.com/leotulipan/transcribe/internal/adapters/config"
    "github.com/leotulipan/transcribe/internal/core/domain"
    "github.com/leotulipan/transcribe/internal/ports"
)

func newSetupCmd(d Deps) *cobra.Command {
    var (
        groq, openai, assembly, eleven, gemini, mistral string
        defaultProvider, defaultLang, ffmpegPath        string
    )
    cmd := &cobra.Command{
        Use:   "setup",
        Short: "Write the on-disk config non-interactively",
        RunE: func(_ *cobra.Command, _ []string) error {
            cfg := d.Config // start from currently loaded state
            if cfg.APIKeys == nil {
                cfg.APIKeys = map[domain.ProviderID]string{}
            }
            apply := func(id domain.ProviderID, v string) {
                if v != "" {
                    cfg.APIKeys[id] = v
                }
            }
            apply(domain.ProviderGroq, groq)
            apply(domain.ProviderOpenAI, openai)
            apply(domain.ProviderAssemblyAI, assembly)
            apply(domain.ProviderElevenLabs, eleven)
            apply(domain.ProviderGemini, gemini)
            apply(domain.ProviderMistral, mistral)
            if defaultProvider != "" {
                cfg.DefaultProvider = domain.ProviderID(defaultProvider)
            }
            if defaultLang != "" {
                cfg.DefaultLanguage = defaultLang
            }
            if ffmpegPath != "" {
                cfg.FFmpegPath = ffmpegPath
            }
            store := config.New()
            if err := store.Save(cfg); err != nil {
                return err
            }
            fmt.Println("wrote", store.Path())
            _ = ports.Config{} // keep import if linter complains
            return nil
        },
    }
    cmd.Flags().StringVar(&groq, "groq-key", "", "Groq API key")
    cmd.Flags().StringVar(&openai, "openai-key", "", "OpenAI API key")
    cmd.Flags().StringVar(&assembly, "assemblyai-key", "", "AssemblyAI API key")
    cmd.Flags().StringVar(&eleven, "elevenlabs-key", "", "ElevenLabs API key")
    cmd.Flags().StringVar(&gemini, "gemini-key", "", "Gemini API key")
    cmd.Flags().StringVar(&mistral, "mistral-key", "", "Mistral API key")
    cmd.Flags().StringVar(&defaultProvider, "default-provider", "", "Default provider id")
    cmd.Flags().StringVar(&defaultLang, "default-language", "", "Default language (ISO-639-1)")
    cmd.Flags().StringVar(&ffmpegPath, "ffmpeg-path", "", "Path to ffmpeg executable (empty = PATH lookup)")
    return cmd
}
```

- [ ] **Step 3: Build and run a smoke test on the help output**

```bash
go build -o bin/transcribe.exe ./cmd/transcribe
./bin/transcribe.exe --help
./bin/transcribe.exe providers --help
./bin/transcribe.exe setup --help
```

Expected: subcommands listed, flags present.

- [ ] **Step 4: Commit**

```bash
git add internal/delivery/cli/providers.go internal/delivery/cli/setup.go
git commit -m "feat(cli): providers + setup subcommands"
```

---

## Phase M — End-to-end smoke test + build

### Task M1: CLI smoke test against a built binary

**Files:**

- Create: `tests/integration/cli_test.go`

- [ ] **Step 1: Write the smoke test**

```go
//go:build integration

package integration_test

import (
    "encoding/json"
    "os"
    "os/exec"
    "path/filepath"
    "runtime"
    "testing"

    "github.com/stretchr/testify/require"
)

// Build the binary once and exercise it.
func TestCLI_JSONMode_HelpExits0(t *testing.T) {
    bin := buildBinary(t)
    out, err := exec.Command(bin, "--help").CombinedOutput()
    require.NoError(t, err, string(out))
    require.Contains(t, string(out), "transcribe")
}

func TestCLI_JSON_RejectsMissingApiKey(t *testing.T) {
    if os.Getenv("TRANSCRIBE_GROQ_KEY") != "" {
        t.Skip("real GROQ key present — this test expects missing key")
    }
    bin := buildBinary(t)
    sample := mustTestdata(t, "short-sample.mp3")
    cmd := exec.Command(bin, "transcribe", "--json", "--api", "groq", "--output", "text", sample)
    cmd.Env = append(os.Environ(), "TRANSCRIBE_GROQ_KEY=")
    out, _ := cmd.CombinedOutput()
    var got map[string]any
    require.NoError(t, json.Unmarshal(out, &got), string(out))
    require.Equal(t, "error", got["status"])
}

func buildBinary(t *testing.T) string {
    t.Helper()
    dir := t.TempDir()
    name := "transcribe"
    if runtime.GOOS == "windows" {
        name += ".exe"
    }
    bin := filepath.Join(dir, name)
    cmd := exec.Command("go", "build", "-o", bin, "../../cmd/transcribe")
    out, err := cmd.CombinedOutput()
    require.NoError(t, err, string(out))
    return bin
}

func mustTestdata(t *testing.T, name string) string {
    t.Helper()
    wd, err := os.Getwd()
    require.NoError(t, err)
    return filepath.Join(wd, "..", "..", "testdata", name)
}
```

- [ ] **Step 2: Run with the integration tag**

```bash
go test -tags integration ./tests/integration/...
```

Expected: PASS (the binary builds, `--help` succeeds, missing-key flow returns a JSON error).

- [ ] **Step 3: Commit**

```bash
git add tests/integration/cli_test.go
git commit -m "test(integration): cli smoke against built binary"
```

---

### Task M2: Version injection + build script

**Files:**

- Create: `scripts/build.ps1`
- Modify: `BUILD.md`

- [ ] **Step 1: Write `scripts/build.ps1`**

```powershell
param(
    [string]$Version = "dev",
    [string]$Out = "bin/transcribe.exe"
)
$ErrorActionPreference = "Stop"
if (-not $Version -or $Version -eq "dev") {
    try { $Version = (git describe --tags --always 2>$null) } catch {}
    if (-not $Version) { $Version = "dev" }
}
Write-Host "Building $Out (version=$Version)"
go build -ldflags "-X main.version=$Version" -o $Out ./cmd/transcribe
```

- [ ] **Step 2: Append to `BUILD.md`**

```markdown

## Reproducible build

```powershell
./scripts/build.ps1                          # version from git describe
./scripts/build.ps1 -Version v1.0.0          # explicit version
```

`./bin/transcribe.exe --version` will report the embedded version.

```

- [ ] **Step 3: Smoke-test the script**

```powershell
./scripts/build.ps1 -Version smoke-test
./bin/transcribe.exe --version
```

Expected: prints `transcribe version smoke-test`.

- [ ] **Step 4: Commit**

```bash
git add scripts/ BUILD.md
git commit -m "build: powershell build script + version injection"
```

---

## Self-review

Run these at the end of executing the plan, before declaring done:

- [ ] `go vet ./...` — clean
- [ ] `go test -race ./...` — all pass
- [ ] `go build -o bin/transcribe.exe ./cmd/transcribe` — succeeds
- [ ] With a Groq key set: `bin/transcribe.exe transcribe --api groq --output text,srt,davinci_srt testdata/short-sample.mp3` produces three files next to the input
  - [ ] note: use api keys from C:\Users\leona\.transcribe\.env
- [ ] With a Groq key set: `bin/transcribe.exe transcribe --json --api groq --output text testdata/short-sample.mp3` emits a single JSON object on stdout
- [ ] Re-run the same command with `--use-cache` — second run skips the API call
- [ ] Move/rename the source file mid-job (Ctrl-C) and confirm `.transcribe-tmp/<basename>/` retains the completed intermediate; re-run resumes from it.

Plan 1 is complete when all of the above pass.
