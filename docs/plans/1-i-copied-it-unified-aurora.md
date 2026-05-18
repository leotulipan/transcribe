# GUI UX + ffmpeg fix + model discovery

## Context

After deploying the Go binary to `%USERPROFILE%\.local\bin` and running it
against real audio, five issues surfaced:

1. **Discoverability** — unclear where the GUI writes its config (and whether
   it reads from `%USERPROFILE%\.transcribe\` like the Python tool did).
2. **ffmpeg breaks transcription** — when the user enters the WinGet shim
   directory (`%LOCALAPPDATA%\Microsoft\WinGet\Links\`) as the ffmpeg path,
   `exec.CommandContext` fails because that string is a directory, not an
   executable. Auto-discovery doesn't try the WinGet location either.
3. **Restart required after Settings save** — current dialog literally says
   "Restart transcribe to pick up new keys."
4. **GUI is single-file only** — no folder selection, no drag/drop, so the
   CLI's batch loop isn't reachable from the GUI.
5. **Model lists are stale-by-design** — every adapter returns a hardcoded
   slice. ElevenLabs in particular has more usable models than what's hard-coded.

Goal: unblock transcription (item 2 is a real bug), then ship the UX wins,
then close the model-discovery gap.

## Answer to "where is config saved?"

**`%LOCALAPPDATA%\transcribe\config.toml`** (see
`internal/adapters/config/tomlstore.go:53-63` `defaultPath`). The Go port does
NOT load from `%USERPROFILE%\.transcribe\` the way the Python tool did. The
existing repo-local `.transcribe.toml` walk-up (`findLocalConfig`) already
handles per-checkout overrides; adding a third user-level path would create
precedence ambiguity. Decision: document the current path in the GUI dialog
header + README, do not add the alternate lookup.

## Phases

| Phase | Items | Why this order |
|---|---|---|
| **A. ffmpeg fix** | #2 | Real bug, breaks transcription. Smallest blast radius. Ship alone. |
| **B. Settings UX** | #1 + #3 | Doc surfaces in the same dialog that gains no-restart save. |
| **C. Batch + drag/drop** | #4 | Independent. Depends on extracting a folder-enumerate helper. |
| **D. Model discovery** | #5 | Largest scope, schema change, new CLI subcommand. |
| **E. Version + 0.9.0 release** | semver | Cuts the first proper release once A–D are merged. |

---

## Phase A — ffmpeg resolution

**New file:** `internal/adapters/audio/resolve.go`
```go
func ResolveBinary(userPath, binName string) (string, error)
var ErrFFmpegNotFound = errors.New("ffmpeg/ffprobe binary not found")
```
Algorithm (in order; first success wins):
1. **User-provided path:** if `userPath` non-empty → if `os.Stat` shows a
   directory, append `binName` (`.exe` on Windows). Final `Stat` must be a
   regular file. If valid, return canonical absolute path.
2. **Well-known shim locations:** check
   `%LOCALAPPDATA%\Microsoft\WinGet\Links\<binName>.exe` (and any other shim
   path we discover — chocolatey `C:\ProgramData\chocolatey\bin`, scoop
   `%USERPROFILE%\scoop\shims`, etc.). Skip on non-Windows.
3. **`exec.LookPath(binName)`** — the standard library's PATH walk
   (respects PATHEXT on Windows).
4. **Explicit PATH walk** — fallback for when `LookPath` returns no match
   but the binary is reachable via an unusual PATH entry (e.g. quoted entry,
   trailing whitespace, case-sensitivity quirks Go's stdlib misses). Iterate
   `filepath.SplitList(os.Getenv("PATH"))`, try each `<dir>/<binName>` and
   `<dir>/<binName>.exe` on Windows. Return the first that `Stat`s as a
   regular file.
5. Return `ErrFFmpegNotFound` with all attempted locations in the message.

**Touch:**
- `internal/adapters/audio/ffmpeg.go:17-33` — `New(...)` calls
  `ResolveBinary` for both ffmpeg + ffprobe. CLI path: return error.
  GUI path: emit a `log` warning and fall back to auto-discover, never panic.
- `internal/delivery/cli/setup.go` — normalize FFmpegPath via `ResolveBinary`
  before writing TOML; refuse save with clear message if it resolves to nothing.
- `internal/delivery/gui/settings.go:~67` — same on save; populate the entry
  with the canonical resolved path so the user sees what was stored.
- `internal/adapters/config/tomlstore.go` `Save` — best-effort normalize on
  write (warn-and-store-canonical, don't reject).

**Tests:** `internal/adapters/audio/resolve_test.go` table cases (empty, dir,
file, missing, .exe-less Windows path) using `t.TempDir()` + fake binaries
created via `os.WriteFile`. Existing transcode integration tests cover the
happy exec path.

---

## Phase B — Settings UX (no-restart + doc surface)

**Touch:**
- `cmd/transcribe-gui/main.go:23-28` — keep startup `Load` + `BuildService`.
  Pass a pointer/handle to the GUI so it can re-init.
- `internal/delivery/gui/deps.go` (or wherever `gui.Deps` lives) — add:
  ```go
  func (d *Deps) Reload() error  // Load config, BuildService, atomically
                                  // swap d.Service and d.Config under RWMutex.
  ```
  Reads of `d.Service` from job submission paths take `RLock` for the duration
  of "capture the pointer + hand off to background goroutine", then release.
  `Reload` takes `Lock`. **In-flight jobs keep the old service.** Never swap
  mid-job.
- `internal/delivery/gui/settings.go` —
  - Open handler reloads config from disk before populating fields (so the
    dialog shows whatever was last saved, not the startup snapshot).
  - Save handler calls `d.Reload()` after `SaveConfig`.
  - Replace line 67 "Restart..." with "Settings applied."
  - Show resolved config path in the dialog header (e.g.
    `"Config: %LOCALAPPDATA%\transcribe\config.toml"`) — covers item #1's
    "where is it?" question visually.

**Docs:**
- `README.md` + `BUILD.md` — add a "Configuration" section: file locations,
  env-var precedence, repo-local `.transcribe.toml` for per-checkout keys.
  Explicitly note this differs from the Python tool's `%LOCALAPPDATA%\audio_transcribe\.env`.

**Tests:** `gui/deps_test.go` unit test on `Reload` with temp config dir +
fake provider registry. GUI dialog logic remains manual QA per
`docs/superpowers/plans/4-gui-smoke-checklist.md`.

---

## Phase C — Batch + drag/drop

**New helper:** `internal/core/services/enumerate.go`
```go
func EnumerateAudioFiles(root string) ([]string, error)
```
- If `root` is a file: return `[root]`.
- If a dir: walk recursively, include
  `mp3,wav,m4a,flac,ogg,opus,oga,mp4,mkv,mov,avi,webm`,
  skip hidden files, return sorted list.
- Extension constants currently live in `internal/delivery/cli/transcribe.go`
  (find via grep) — move them into one place this helper imports.

**Touch:**
- `internal/delivery/cli/transcribe.go` `runTranscribe` — replace inline
  enumeration with `EnumerateAudioFiles`.
- `internal/delivery/gui/mainwindow.go:115-122` — add "Select Folder" button
  next to existing Browse; uses `dialog.ShowFolderOpen`. **On selection,
  store the directory path in the entry as-is — do NOT enumerate yet.** The
  user sees the chosen folder path; enumeration happens lazily.
- Submit ("Start") handler: when the entry value is a directory, call
  `EnumerateAudioFiles` *at click time*, populate the progress bar with the
  resulting file count, then iterate **sequentially** (one job at a time,
  progress shown per file). This means folder selection is cheap and the
  user can still edit the path before kicking off the batch.
- Same file — in constructor, `window.SetOnDropped(func(_ fyne.Position,
  uris []fyne.URI) { ... })`. Accept first URI (file or dir), populate path
  entry on Fyne main thread (`fyne.Do`), refresh.

**Tests:** `enumerate_test.go` — tempdir tree with mixed extensions, nested
dirs, hidden files. GUI drop handler tested manually.

---

## Phase D — Model discovery

**New optional interface:** `internal/ports/discovery.go`
```go
type ModelDiscoverer interface {
    DiscoverModels(ctx context.Context) ([]string, error)
}
```
Adapters that don't support a listing endpoint simply don't implement it.

**Per-adapter implementation** — new `discover.go` in each of:
`groq, openai, mistral, gemini, elevenlabs`. Each reuses the existing
CheckKey HTTP plumbing, parses provider-specific JSON, returns a sorted
unique slice.

- **groq / openai / mistral** — `GET /v1/models`, Bearer auth, response
  `{"data":[{"id":"..."}]}`. Extract every `id`.
- **gemini** — `GET /v1beta/models`, `x-goog-api-key` header, response
  `{"models":[{"name":"models/..."}]}`. Strip the `models/` prefix.
- **elevenlabs** — `GET /v1/models`, `xi-api-key` header. Response:
  ```json
  [{"model_id":"...","name":"...","can_do_text_to_speech":true,
    "can_do_voice_conversion":true, ...}]
  ```
  No `can_do_speech_to_text` flag in the schema. **First implementation:
  return every `model_id`** and let the user pick. If a future API revision
  exposes an STT flag (or scribe_v1 shows up there), filter then. Document
  this in the package comment so it's not forgotten.

**Skip** `assemblyai` (no listing endpoint known); document this gap in its
package comment. Its `Models()` stays hardcoded.

**Schema:** extend `fileShape` (`internal/adapters/config/tomlstore.go:39-44`):
```go
DiscoveredModels map[string][]string `toml:"discovered_models"`
```
Stored under `[discovered_models]` table: `groq = [...]`, `openai = [...]`, etc.

**Service merge:** in `internal/core/services/service.go:37-43` `ListModels`,
prefer `cfg.DiscoveredModels[id]` when non-empty, else fall back to
`provider.Models()`. **Replace, not union** — user's per-user answer.

**New CLI command:** `internal/delivery/cli/discover_models.go` —
`transcribe discover-models [--provider id] [--json]`. Iterates providers
that implement the optional interface; writes results back to TOML.
JSON envelope:
```json
{
  "saved_to": "...config.toml",
  "results": [
    {"provider":"groq","count":42,"models":[...],"error":null},
    ...
  ]
}
```
Wire into `internal/delivery/cli/root.go` next to `test-keys`.

**GUI:** small refresh icon next to the model dropdown in `mainwindow.go`
**per-provider** (per user choice). Click → call service discover →
persist via `SaveConfig` → reload dropdown. Settings dialog gets no
all-at-once button.

**Tests:**
- Unit per adapter with `httptest.Server` returning canned JSON (extend the
  existing `client_test.go` pattern).
- Integration tagged tests hit real APIs (skip when key missing, same
  pattern as `CheckKey` integration tests).
- CLI command tested via existing CLI test harness.
- Service `ListModels` unit test: with and without `DiscoveredModels`.

---

---

## Phase E — Semver + 0.9.0 release

**Problem:** `transcribe --version` currently reports things like
`transcribe version dev-checkkey` (or whatever ad-hoc `-Version` string was
passed to `scripts/build.ps1`). That's not informative and is not a valid
semver string.

**Approach:**
- All future versions follow `MAJOR.MINOR.PATCH` (semver 2.0.0). Pre-1.0
  releases use the `0.x.y` range freely; breaking changes bump MINOR pre-1.0.
- **Cut `v0.9.0` as the first proper release** after Phases A–D land.
  Reserves `1.0.0` for the moment we feel public-API stable (CLI flags,
  config schema, GUI surface).
- Tag format: `vMAJOR.MINOR.PATCH` (lowercase `v`), e.g. `v0.9.0`.
- Build embeds the version via `-ldflags "-X main.version=$Version"`
  (already wired in `scripts/build.ps1`). With proper tags, `git describe
  --tags --always` produces `v0.9.0` on the tagged commit and
  `v0.9.0-3-gabc1234` on dirty / post-tag commits.

**Touch:**
- `scripts/build.ps1` — accept the version as-is from `git describe`; reject
  anything that doesn't look like `vNN.NN.NN` (or `vNN.NN.NN-...`) unless
  `-Version` is explicitly overridden to a non-semver string (allows ad-hoc
  smoke builds). Print a warning on non-semver `-Version` input.
- `cmd/transcribe/main.go:21` and `cmd/transcribe-gui/main.go` (find the
  matching `var version = "dev"`) — keep default as `"dev"` for unbuilt
  `go run` invocations, but document that release builds always pass an
  explicit version.
- `CHANGELOG.md` — add `[0.9.0]` heading, summarize Phases A–D. Bump
  `[Unreleased]` accordingly.
- `BUILD.md` — add a "Release" subsection documenting the tag + build
  workflow: bump CHANGELOG, `git tag -a v0.9.0 -m "..."`, push tag,
  run `scripts/build.ps1`, ship `bin/transcribe.exe` and
  `bin/transcribe-gui.exe`. Optionally a GitHub release with the zips.

**Verification:**
```
git tag -a v0.9.0 -m "Initial Go port release"
powershell.exe -File scripts/build.ps1
.\bin\transcribe.exe --version       # expect: transcribe version v0.9.0
.\bin\transcribe-gui.exe --version   # same
```

**Note:** `pyproject.toml` version stays separate (Python tool still has
its own release line in `python/` eventually). Go binary version is
authoritative for the Go port.

---

## Critical files

- `internal/adapters/audio/ffmpeg.go` — ffmpeg validation entry point.
- `internal/adapters/audio/resolve.go` — **new**, path resolution.
- `internal/adapters/config/tomlstore.go` — schema + normalize on save.
- `internal/delivery/gui/settings.go` — settings dialog (restart removal).
- `internal/delivery/gui/mainwindow.go` — folder picker + drag/drop +
  model refresh button.
- `internal/core/services/enumerate.go` — **new**, batch file enumeration.
- `internal/core/services/service.go` — `ListModels` merge logic.
- `internal/delivery/cli/setup.go` — ffmpeg validation on save (CLI).
- `internal/delivery/cli/discover_models.go` — **new**, CLI command.
- `internal/ports/discovery.go` — **new**, optional interface.
- `internal/adapters/api/{groq,openai,mistral,gemini,elevenlabs}/discover.go` — **new**.
- `scripts/build.ps1`, `cmd/transcribe/main.go`, `cmd/transcribe-gui/main.go`,
  `CHANGELOG.md`, `BUILD.md` — Phase E semver + release wiring.

## Reusable existing utilities

- HTTP/auth scaffolding in each adapter's `CheckKey` (e.g.
  `internal/adapters/api/groq/client.go:58-79`) — copy-adapt for `DiscoverModels`.
- `internal/integration.Key(t, providerID)` for integration tests.
- `delivery.BuildService` for `Deps.Reload`.
- `tomlstore.Load`'s 3-layer merge (user TOML → repo-local → env) — already
  works correctly; discovery results land in the user TOML layer.

## Verification

**Phase A (ffmpeg fix):**
```
go test ./internal/adapters/audio/...
.\bin\transcribe.exe setup     # type a directory path, confirm it's normalized
.\bin\transcribe.exe transcribe --api elevenlabs --output text <real-mp4>
```
Manual: open GUI settings, paste WinGet Links dir, save, confirm the entry
updates to the canonical `\ffmpeg.exe` full path.

**Phase B (settings UX):**
```
go test ./internal/delivery/gui/...
```
Manual: change API key in GUI, save, immediately run a transcription without
restarting — must use the new key. Reopen Settings — must show the new key.

**Phase C (batch + drag/drop):**
```
go test ./internal/core/services/... -run Enumerate
```
Manual: drop a folder onto the GUI window. Confirm the entry fills. Run
transcribe — must process every file sequentially with per-file progress.

**Phase D (model discovery):**
```
go test ./internal/adapters/api/groq/... -run Discover
go test -tags=integration -run Discover ./internal/adapters/api/...
.\bin\transcribe.exe discover-models                       # all providers
.\bin\transcribe.exe discover-models --provider groq --json
```
Manual: open GUI, click refresh next to a provider dropdown, confirm the
dropdown re-populates with the live list and the TOML file shows the
`[discovered_models]` block.

**Phase E (release):**
```
git tag -a v0.9.0 -m "Initial Go port: CheckKey, batch GUI, model discovery, ffmpeg fix"
powershell.exe -File scripts/build.ps1
.\bin\transcribe.exe --version       # expect: transcribe version v0.9.0
.\bin\transcribe-gui.exe --version
```

**Full suite after each phase:**
```
go test ./...
go test -tags=integration ./...     # requires .transcribe.toml keys
```

## Risks / cross-cutting

- **`d.Service` swap concurrency** — RWMutex pattern documented above is the
  only correct way. Do not skip the lock.
- **TOML write atomicity** — confirm `tomlstore.Save` uses write-temp +
  rename; both `discover-models` and GUI settings now write. If not,
  add atomic-rename. Document last-writer-wins for the rare concurrent case.
- **Optional interface assertion** — `if d, ok := p.(ports.ModelDiscoverer); ok`
  at call sites. Never panic on missing implementations.
- **Fyne drag/drop** — `SetOnDropped` is desktop-only; no-op on mobile.
  Add a code comment.
- **Resolved ffmpeg path UI update** — must happen on Fyne main thread
  (`fyne.Do` or `widget.Refresh`) to avoid race.
- **Stale discovered models** — if a model is removed upstream, the user's
  stored list won't know until they refresh. Acceptable trade-off; the
  `transcribe` command will surface API errors at submit time anyway.
