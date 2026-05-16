# Python → Go Port Playbook

A field-tested checklist for porting a Python CLI tool to a single-binary Go
implementation. Distilled from the `transcribe` port (6 providers, 3 output
formats, ffmpeg pipeline, 3 UIs, Windows-only v1). Use as a brainstorming seed
and design reference for the next port.

The audience is a future "you" — or a future Claude Code session you're seeding.
Read top-to-bottom on a new port, copy-paste relevant sections into the
brainstorming prompt, skip the parts your project doesn't need.

---

## 0. Pre-flight (do these BEFORE any brainstorming)

1. **Toolchain check** — verify availability *up front*, not when a phase blocks on them:
    - Go 1.22+ (`go version`)
    - `gcc` (only if any plan needs CGO — Fyne, raylib, sqlite-cgo, etc.)
    - External CLIs the Python tool shells to (ffmpeg, imagemagick, etc.)
    - On Windows: each tool reachable in a *fresh* shell, not just the one
      that did the install. Subagent shells don't inherit a freshly-installed
      PATH — persist via `~/.bashrc` (Git Bash) or system-PATH + new session.

2. **Isolation** — create a git worktree off the Python branch (e.g.
   `worktree-go-port`). Python stays untouched on main until the Go branch
   reaches parity; one final PR removes the Python tree.

3. **Module path** — `github.com/<user>/<project>` or your private path. Pick
   once, never rename.

4. **Inventory the Python tool** — read the existing README, CLAUDE.md, and
   demo/howto docs (or generate a demo with `uvx showboat` first). You need a
   clear "what does this thing actually do, end-to-end" before scoping.

---

## 1. Brainstorming — questions to answer (with default recommendations)

| Question | Default answer (justified by transcribe) | Why |
|---|---|---|
| **MVP scope** | Core parity, defer polish | "All providers + headline output formats; defer diarization, niche formats, batch templates." Slim MVP feels appealing but you'll regret missing the killer feature. Full parity bloats v1. |
| **External media handling** | Shell out to ffmpeg/etc. | Smaller binary, full format coverage. Embedding the binary is 30–80 MB bloat per platform plus AV false positives on Windows. Pure-Go libs lack codec/demuxer coverage. |
| **CLI→TUI escalation** | Escalate when required input is missing in non-JSON mode | "Forgot a flag, fall into a wizard." Native to non-technical users; pure-CLI users supply everything via flags and never see the TUI. |
| **State compatibility with Python** | Fresh start (new config location, new cache schema) | Locks you to Python quirks forever otherwise. Users re-enter API keys once; existing caches ignored. Cleanest design. |
| **Zero-args default UI** | GUI if a GUI ships; else TUI; never CLI | A user who runs the bare exe is asking for help, not a usage message. |
| **Repo layout** | Worktree off the Python branch; Python retires when Go reaches parity | Cleanest history. Subdirectory layout (`go/` next to `python/`) works but creates dual-build CI complexity. |
| **Service input port shape** | `Submit(ctx, req) → Job` with `Job.Progress() <-chan Event` | Service owns the goroutine; UIs poll the handle. Cleaner than synchronous-method-with-channel-arg for batch mode and long-lived GUI sessions. Callback sinks are wrong-threaded for both Bubble Tea and Fyne. |
| **Cleanup policy for temp/intermediate files** | Keep on transient errors; delete on success or permanent error | Lets retries resume from a complete intermediate. Partials always deleted. |
| **Intermediate file location** | `<input_dir>/.transcribe-tmp/<basename>/`, falling back to `os.TempDir()` if read-only | Source-adjacent enables resume across runs; `os.TempDir()` is the safe fallback. |
| **Env var names** | Canonical SDK names (`GROQ_API_KEY`, `OPENAI_API_KEY`) | Users' existing `.env` files already use these. No custom prefix. |
| **macOS/Linux build** | Windows-only v1; scaffolding for v2 | One platform validates the architecture. Cross-platform CGO + macOS notarization is its own project. |
| **Distribution** | Single binary per platform, plus a `<tool>-gui.exe` Windows-only flavor with `-H windowsgui` | The GUI flavor stops Explorer launches from popping a console. |

If your project differs in nature (server vs CLI, embedded media vs none, etc.),
revisit each row; the structure of the question still applies.

---

## 2. Architecture defaults — hexagonal layout

```
.
├── go.mod / go.sum
├── cmd/
│   ├── <tool>/           # main entrypoint — parses --ui, dispatches
│   └── <tool>-gui/       # Windows -H windowsgui flavor (same code, no console)
├── internal/
│   ├── core/
│   │   ├── domain/       # plain structs + sentinel errors. No external imports.
│   │   └── services/     # orchestration + pipeline. Imports domain + ports only.
│   ├── ports/            # interfaces. Imports domain only.
│   ├── adapters/         # one subpackage per external concern.
│   │   ├── api/          # remote-service clients (one subpkg per provider)
│   │   │   └── internal/retry/   # shared HTTP retry helper
│   │   ├── <external>/   # ffmpeg, db, fs, etc.
│   │   ├── config/       # config store
│   │   ├── cache/        # result + intermediate caches
│   │   ├── format/       # output writers
│   │   └── logging/      # log/slog wrapper
│   └── delivery/         # composition root + UIs
│       ├── wire.go       # the *only* place that imports adapters
│       ├── cli/
│       ├── tui/
│       └── gui/
├── docs/
│   ├── superpowers/
│   │   ├── specs/        # design specs
│   │   └── plans/        # implementation plans
│   └── python-to-go-playbook.md   # this file
├── testdata/             # fixtures (small media + golden outputs + sample JSON)
├── tests/
│   ├── integration/      # built-binary smokes, build tag `integration`
│   └── scripts/          # load_env.sh + load_env.ps1 for real-API runs
└── scripts/
    └── build.ps1         # Windows build (both flavors, version injection)
```

**Layer rules (enforce in code review):**

- `internal/core/domain` imports only stdlib.
- `internal/core/services` imports `core/domain` + `ports`. **Never** an adapter, **never** `fmt.Print*`, **never** UI code.
- `internal/ports` imports only `core/domain`.
- `internal/adapters/*` imports `ports` + `core/domain` + its own external deps. **Never** another adapter, **never** `core/services`, **never** `delivery`.
- `internal/delivery/*` is the **only** place that imports concrete adapters.

---

## 3. Recurring design patterns (port them verbatim where applicable)

### 3.1 Job-handle + progress channel

```go
type Service interface {
    Submit(ctx context.Context, req Request) (Job, error)
}
type Job interface {
    ID() string
    Progress() <-chan ProgressEvent   // closed when the job ends
    Wait() (*Result, error)            // safe to call repeatedly
    Cancel()                           // idempotent
}
```

- Service spawns one goroutine per Submit.
- Progress channel is buffered (cap ~32); drops events if a slow UI doesn't keep up — the channel never blocks the pipeline.
- `Wait` caches `(result, err)` so it's safe for multiple callers.

### 3.2 Capability check before expensive work

Provider declares per-model capabilities (e.g., `WordTimestamps`, `AcceptedInputs`).
Service rejects incompatible request/format combos *before* doing any I/O.
Return a typed `ErrIncompatible` with the offending field so the UI can render
"your selection conflicts: X + Y — pick A or switch to B".

### 3.3 Copy-first media preparation (when sending audio/video upstream)

Decision tree:

1. **As-is** — input codec/container accepted by the API *and* under size limit → no work.
2. **Stream copy** — codec accepted but container isn't (mp4-wrapped AAC → m4a) → `ffmpeg -c:a copy`. No re-encode, no quality loss.
3. **Transcode** — fall through to lossy/lossless re-encode targeted at the cheapest accepted codec.
4. **Chunk** — only if (3) still exceeds the API's byte limit.

Every step writes to a `*.partial` filename and atomically renames to the final
name on success. A crash leaves a `*.partial` that the cleanup pass deletes.

### 3.4 Two-layer cache

- **Result cache** — sidecar JSON next to input (`<basename>.<tool>.<provider>.json`). Schema-versioned envelope. Cache hit short-circuits the entire pipeline.
- **Intermediate cache** — `<input_dir>/.<tool>-tmp/<basename>/` holding extracted/transcoded audio with a `<file>.meta.json` sidecar recording `{operation, source_size, source_mtime, target_codec, provider, model, budget}`. On the next run, lookup matches and skips re-extraction.

### 3.5 Typed errors with Unwrap

```go
var (
    ErrConfigMissing   = errors.New("config error")
    ErrProviderMissing = errors.New("provider not configured")
    ErrCanceled        = errors.New("canceled")
)
type ErrIncompatible struct { Provider, Model, Format, Reason string }   // value receiver
type ErrProvider struct { Provider; StatusCode int; Retryable bool; Cause error }  // pointer receiver, Unwrap() returns Cause
```

Exit-code mapping uses `errors.Is` / `errors.As` over this catalog:

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | internal / unexpected |
| 2 | usage error (bad flags, missing args, capability mismatch) |
| 3 | config error (missing key, missing external tool) |
| 4 | upstream provider error (after retries) |
| 5 | external-tool / media error |
| 130 | canceled via signal |

JSON error payloads carry the same `"code"` so agents can branch on either.

### 3.6 Agent-callable `--json` mode

CLI flag that forces three things:

1. Pure CLI, no escalation. Missing required args → exit 2 with JSON error.
2. Stdout is exclusively JSON. Logs go to stderr.
3. Two schemas:
    - Without `--progress`: one final object `{schema_version, status, result?, error?}`.
    - With `--progress`: JSON Lines stream `{type:"progress",...}` terminated by `{type:"result"|"error",...}`.

Schema lives in **one renderer file**. Change the schema, change exactly one place.

### 3.7 Per-OS config paths via `runtime.GOOS`

```go
func defaultConfigPath() string {
    if runtime.GOOS == "windows" {
        return filepath.Join(os.Getenv("LOCALAPPDATA"), "<tool>", "config.toml")
    }
    home, _ := os.UserHomeDir()
    return filepath.Join(home, ".<tool>", "config.toml")
}
```

TOML over `.env` — typed, comments, no quoting ambiguity. `pelletier/go-toml/v2` is the standard library.

### 3.8 Canonical SDK env var names

Use each vendor's own env var name (`OPENAI_API_KEY`, `GROQ_API_KEY`, etc.), not
a custom prefix. Users have `.env` files that work with the official SDKs;
yours should consume the same names. Defensive: trim whitespace + trailing
CR in `Load()` because Windows `.env` files often have CRLF.

### 3.9 Delivery composition root (single wire site)

```go
// internal/delivery/wire.go
func BuildService(cfg ports.Config, log ports.Logger) (ports.TranscribeService, error) {
    // instantiate every adapter here, exactly once
    // hand the assembled Service to the UI
}
```

UIs see only `ports.TranscribeService`. Swap any adapter by editing this one file.

---

## 4. Implementation strategy

### 4.1 Decompose into shippable plans

Each plan ends with working, testable software. For the transcribe port:

1. **Foundation** — domain + ports + service + one provider + CLI (with `--json`). Ships a working binary for one provider.
2. **Provider expansion** — remaining providers, each ~3-4 tasks. Ships multi-provider CLI.
3. **TUI** — Bubble Tea screens + escalation. Ships interactive terminal UI.
4. **GUI** — Fyne window + threading. Ships desktop app.

Each plan can stop the project on a working milestone.

### 4.2 Plan structure (per task)

- File map first (what files this task touches, one responsibility each).
- TDD steps: write failing test → run to confirm failure → minimal implementation → run to confirm pass → commit.
- **Verbatim code** in every step. Plans with `TODO: implement` rot fast.
- One commit per task with a semantic message.
- Account for adapter behavior under test (skip-if-no-binary helpers, `httptest.NewServer` for HTTP, golden files for formatters).

### 4.3 Subagent dispatch (when using Claude Code)

- One subagent per phase (5–10 tasks).
- Two-stage review *after each implementer*: spec compliance, then code quality. Skip both stages only for pure type/interface definitions (no logic to review).
- Model tiers:
    - **Haiku** — mechanical setup (mod init, ignore files, BUILD.md).
    - **Sonnet** — type/interface/adapter/service implementation.
    - **Sonnet/Opus** — final all-up review of the whole branch.
- After each phase, verify in-line in the dispatcher session before moving on (saves subagent budget vs review subagents for trivial phases).
- **Always** end with one final reviewer subagent over the whole branch.

### 4.4 Real-API smoke before declaring complete

At least one provider, end-to-end against a real key. Done via the user's
existing `.env` plus a small helper script that strips CRLF/whitespace. The
remaining providers can ship with "tested via mock httptest server, real API
not verified" flagged as a known caveat.

---

## 5. Hard constraints (encode in CLAUDE.md, enforce in final review)

- **No `fmt.Print*` in `internal/core/` or `internal/adapters/`.** All output goes through `ports.Logger` or return values.
- **No callbacks across goroutine boundaries.** Async work uses goroutines + channels; UIs marshal to their own event loops (Bubble Tea via `tea.Cmd`, Fyne via `fyne.Do`).
- **Single binary per platform.** No external runtime deps the user has to install (other than declared external CLIs like ffmpeg).
- **Hexagonal import rules** (see §2 layer rules).
- **No `cd <our-cwd> && ...` in any shell command.** It triggers Windows approval prompts and is always redundant — see `CLAUDE.md`'s Shell Commands section.

---

## 6. Gotchas catalog

| # | Gotcha | Fix |
|---|---|---|
| 1 | **Windows CRLF breaks golden tests.** Git's `autocrlf=true` rewrites checked-out `.golden` files, golden byte comparison fails. | Add `*.golden text eol=lf` to `.gitattributes` *before* the first golden lands. |
| 2 | **`ffmpeg` can't detect output format from a `*.partial` extension.** | Pass `-f <format>` explicitly: `mp3→mp3, flac→flac, wav→wav, m4a→ipod, ogg→ogg`. |
| 3 | **Subagent shells don't inherit a freshly-installed PATH.** Installing Go via winget doesn't make `go` visible in already-spawned shells. | Persist additions in `~/.bashrc`. |
| 4 | **`.env` files on Windows have CRLF + trailing whitespace.** Loaders strip `=` but leave `\r`, breaking HTTP `Authorization` headers (`invalid header field value`). | Strip `\r` before splitting on `=`; trim trailing whitespace from values. Trim defensively in `config.Load()` too. |
| 5 | **`go test -race` requires CGO.** | Either install gcc or just run unit tests without `-race`; enable it only when concurrency code matters. |
| 6 | **Provider API drift.** Plan was written against version X; live API is now version Y. | Before writing the client, query the provider's docs via context7 (or fetch the canonical SDK source). Mark known drift as DONE_WITH_CONCERNS rather than guessing silently. |
| 7 | **Fyne needs CGO + a C compiler.** No headless test harness. | Install MSYS2 + `mingw-w64-ucrt-x86_64-gcc`. GUI gets a manual smoke checklist, not unit tests. |
| 8 | **`.gitignore` patterns without `/` match anywhere.** `transcribe` matched `cmd/transcribe/` and hid source files. | Anchor with `/`: `/transcribe`, `/transcribe.exe`. |
| 9 | **Bubble Tea's `tea.WithContext` exists but timing differs across versions.** | Pin a known-good Bubble Tea minor (≥ v1.3); `go mod tidy` to fetch transitive deps from bubbles sub-packages. |
| 10 | **Capability-check semantics for "blank container" in `AcceptedInputs`.** Reading "any container accepted" naively lets `mp4`-wrapped `aac` skip stream-copy and fail. | Blank `Container` = "accept as-is only when codec == container (e.g. mp3/mp3)"; otherwise stream-copy. |
| 11 | **`{Codec: ..., Container: ...}` truth tables get subtle.** Test for as-is vs copy vs transcode paths *per case* table-driven. | One row per decision-tree branch. |
| 12 | **Empty `AudioFile` from a fake adapter still hits `IsTemp=true`.** Chunk cleanup leaks if temp files aren't tracked. | Track every `Complete=true` intermediate in `tempFiles`; deferred cleanup pass iterates that list. |

Add to this table as new ports surface new gotchas.

---

## 7. Verification checklist

Before declaring a plan complete, mechanically check:

- [ ] `go vet ./...` — clean
- [ ] `go test ./...` — all packages pass
- [ ] `go test -tags integration ./...` against at least one real provider key — passes (or test is documented as deferred)
- [ ] Built binary launches and `--help` exits 0
- [ ] `--json` mode emits valid JSON, exit codes match the documented table
- [ ] No `fmt.Print*` in `internal/core/` or `internal/adapters/` (grep)
- [ ] No `cd <our-cwd>` patterns in any committed script
- [ ] Hexagonal layer rules respected (grep adapter imports of services, etc.)
- [ ] If a GUI plan: manual smoke checklist run on the actual desktop
- [ ] Final reviewer subagent ran with no must-fix findings

---

## 8. Anti-patterns observed and avoided

- **`cd <our-cwd> && cmd`** — never (see CLAUDE.md). Costs the user a confirmation click per command on Windows.
- **Pulling in a vendor SDK when `net/http` suffices.** Vendor SDKs go stale, lock you to their abstractions, bloat the binary. Plain `net/http` + a 60-line client per provider is faster to debug.
- **Mocked "integration" tests.** Unit tests with `httptest.NewServer` are fine; calling those "integration tests" hides drift. Real integration tests hit the real API, gated by env vars + `//go:build integration`.
- **One mega-plan.** Splitting into independent shippable plans gives you natural checkpoints; you can stop after any plan with usable software.
- **Backwards-compatibility shims to ease Python→Go migration.** Tempting; never worth it. Users re-enter API keys once and move on.
- **Adding `--no-verify`, feature flags, or commented-out code "for safety."** Just change the code.
- **Documenting WHAT the code does in comments.** Document WHY when non-obvious; otherwise let well-named identifiers carry the meaning.

---

## 9. Suggested first-prompt seed for a new Python→Go port

> I want to port a Python CLI tool (`<name>`, located at `<path>`) to Go using
> the hexagonal architecture pattern. I've ported one tool this way before and
> have a playbook at `docs/python-to-go-playbook.md` — read that first, then
> read the Python tool's `README.md` and `CLAUDE.md` (and `demo.md` if it
> exists). Then start brainstorming via `superpowers:brainstorming` to nail
> down the decisions in §1 of the playbook. Reuse the defaults unless this
> project differs in a documented way. Target a v1 Windows binary; defer
> macOS/Linux to v2 unless I say otherwise. Use git worktrees; Python stays on
> main until Go reaches parity.

Adjust the platform target and "v1 scope" sentence per the new project.

---

## 10. What this playbook does NOT cover

- **Server / daemon ports.** Long-running services have different lifecycle, observability, and config-reload needs.
- **Database-backed tools.** Migration of SQLite/Postgres schemas, ORM choices, transaction patterns — separate playbook.
- **Heavy concurrency / parallelism beyond per-job goroutines.** If your Python tool uses `asyncio`/multiprocessing for fan-out work (e.g., 100 parallel API calls), you need a worker-pool design this playbook doesn't address.
- **WASM / embedded targets.** Go can target wasm and ARM-cortex, but build pipeline + size budget concerns are separate.

If the new port hits any of those, augment this playbook before starting.

---

*Source project: `transcribe` (Python → Go), May 2026. 70-commit branch, 4 plans, all unit tests green, OpenAI real-API smoke verified.*
