# Plan: Close-out the Go Port and Retire the Python Tree

## Context

The Go port (branch `worktree-go-port`) is feature-rich but not at parity with the legacy Python implementation. Before merging the worktree back into `main`, three things must be true:

1. There is a written, scannable plan for the Python features still missing in Go (word-level SRT, `--chars-per-line`, `--padding-start` logic, filler-words CLI, diarization CLI, TUI completion).
2. The Python `tests/` directory has matching `*_test.go` files in the Go tree — real assertions where the Go feature exists, `t.Skip()` stubs where it doesn't. This locks in the contract and gives us a Red/Green target for the parity work.
3. The Python code, the legacy `tests/` directory, and all Python-era docs are removed from the repo. The Go tree becomes the sole source of truth; the move-then-delete commit pair keeps the cut-over visible in `git log`.

After all three land on `worktree-go-port`, the branch fast-forwards (or merges with `--no-ff` if main has moved) into `main`.

## Gap Summary (from Phase 1 exploration)

**Feature gaps (Python → Go):**
| Feature | Python location | Go status | Flags |
| --- | --- | --- | --- |
| Word-level SRT | `utils/formatters.py` (`create_srt` "word" mode) | **Not in Go**; `FormatText`/`FormatSRT`/`FormatDavinciSRT` only | `--word-srt` |
| SRT line/char length | `transcribe_helpers/output_formatters.py:33` (`wrap_text_for_srt`) | **Not in Go**; hard-coded 7 words/block | `--chars-per-line` |
| `--padding-start` | `output_formatters.py:303` (`apply_intelligent_padding`) | `DaVinciOptions.PaddingStart` field exists, **logic missing** in `applyDavinci` | `--padding-start` |
| Filler-words CLI | `transcribe_helpers/text_processing.py:152` | Logic exists in `services/davinci.go`, **CLI flags not wired** | `--filler-words`, `--remove-fillers`, `--filler-lines` |
| Diarization CLI | `utils/api/elevenlabs.py:103`, `utils/api/assemblyai.py` | **Hardcoded `diarize=false`** at `adapters/api/elevenlabs/client.go:140`; CLI not exposed | `--diarize`, `--speaker-labels` |
| Full TUI | `audio_transcribe/tui/{wizard,interactive}.py` (~370 lines) | Scaffolding only at `internal/delivery/tui/` | (interactive mode) |

**Test gaps (Python tests with no Go equivalent):**
- `test_intermediate_files.py` → `internal/core/services/intermediate_cache_test.go` (already partial; expand)
- `test_language_utils.py` → `internal/core/domain/language_test.go` (**missing**; needs Go language domain too — skip-style)
- `test_output_format_fallback.py` → `internal/core/services/fallback_test.go` (**missing**; feature not in Go — skip-style)
- `test_text_processing.py` → `internal/core/services/davinci_text_test.go` (some coverage in `davinci_test.go`; expand)
- `test_audio_processing.py` deeper coverage of size/format limits → expand `internal/adapters/audio/*_test.go`
- `test_cli.py` golden flows (file vs folder, output formats, language opts) → expand `tests/integration/cli_test.go`

## Stage 1 — Write `docs/plans/2-feature-parity-completion.md`

Write **one consolidated plan** covering the six feature gaps above. Structure inside that file:

```
# Feature Parity Completion (Python → Go)

## Context
<one paragraph: why these features matter, who uses them>

## Phase 1 — Wire existing logic to CLI (Easy)
- Filler-words flags         → cmd/transcribe, services/davinci.go
- Padding-start logic        → core/services/davinci.go (applyDavinci)

## Phase 2 — Output format extensions (Medium)
- Word-level SRT             → adapters/format/srt.go + new FormatWordSRT in domain/ids.go
- --chars-per-line wrapping  → adapters/format/srt.go + adapters/format/davinci.go

## Phase 3 — Provider features (Medium)
- Diarization CLI + plumbing → adapters/api/elevenlabs, adapters/api/assemblyai

## Phase 4 — TUI completion (Hard)
- Interactive mode (matches Python tui/interactive.py)
- Setup wizard (matches Python tui/wizard.py)

## Verification
- New CLI flags appear in `go run ./cmd/transcribe --help`
- Golden SRT fixtures updated in internal/adapters/format/testdata/
- Diarized ElevenLabs run produces speaker labels in SRT
- TUI launch path covered by tui/e2e_test.go
```

Each phase lists: critical files to modify, existing helpers to reuse (with paths), and one verification step. **Reuse `services/davinci.go:applyDavinci` for padding logic** rather than building a parallel pipeline.

This plan file gets snapshot-copied to root `ROADMAP.md` at the start of Stage 3 so it survives the `_legacy/` deletion.

## Stage 2 — Port the Python tests to Go (mix: assertions + skips)

For each Python test file with no Go equivalent, create a `*_test.go` file at the matching Go location. The rule: **if the underlying Go feature exists, write real assertions that exercise it; if it doesn't, write the test scaffold with `t.Skip("pending: <feature-name>")` and a comment pointing at the line in the feature parity plan**.

### New / expanded test files

| Create or expand | Style | Why |
| --- | --- | --- |
| `internal/core/domain/language_test.go` | **Skip** — feature not in Go | Mirrors `tests/unit/test_language_utils.py` |
| `internal/core/services/fallback_test.go` | **Skip** — feature not in Go | Mirrors `tests/unit/test_output_format_fallback.py` |
| `internal/adapters/format/word_srt_test.go` | **Skip** — `FormatWordSRT` not in domain yet | Locks contract before Stage 1 lands |
| `internal/adapters/format/srt_charlen_test.go` | **Skip** — wrap-by-chars not implemented | Mirrors `wrap_text_for_srt` cases |
| `internal/core/services/davinci_padding_test.go` | **Skip** — padding logic missing | Mirrors `apply_intelligent_padding` |
| `internal/adapters/api/elevenlabs/diarize_test.go` | **Skip** — diarize hardcoded false | Locks expected speaker-label output |
| `internal/core/services/intermediate_cache_test.go` | **Real assertions** — exists | Expand existing test with temp-file lifecycle cases from `test_intermediate_files.py` |
| `internal/core/services/davinci_text_test.go` | **Real assertions** — exists | Filler/pause cases from `test_text_processing.py` not already covered |
| `internal/adapters/audio/probe_size_test.go` | **Real assertions** — exists | Per-API size-limit cases from `test_audio_processing.py` |
| `tests/integration/cli_golden_test.go` | **Mix** — some flows exist | Golden cases from `test_cli.py`: file vs folder, output-format combos, language flags |

### Fixture handling

Python fixtures (`tests/fixtures/audio_files/*.{wav,m4a,mkv}` + per-provider JSON responses + `expected_outputs/*.{txt,json}`) need to live somewhere reachable by Go tests **before** Stage 3 moves the Python `tests/` to `_legacy/`. Two options, plan picks (a):

(a) Copy the audio + per-provider JSON fixtures into Go-native `testdata/` directories under their consuming test package (e.g. `internal/adapters/api/elevenlabs/testdata/`). Go convention; `go test` picks them up automatically. **This is what we do.**

(b) Keep a shared root `tests/testdata/` and reference it from Go tests by relative path. Rejected: fragile across packages.

### Test runner

After Stage 2, `go test ./...` should compile cleanly and either pass or skip — no failures. Skipped tests count in the report so the gap is visible.

## Stage 3 — Snapshot, move, delete

Three commits, in order:

### Commit 3a: snapshot + move scope manifest
```
chore(legacy): snapshot roadmap + manifest python retirement
```
- Copy `docs/plans/2-feature-parity-completion.md` → `ROADMAP.md` at repo root (verbatim, with a one-line note: "Source plan was `docs/plans/2-feature-parity-completion.md` before the Python retirement on <date>").
- No other moves yet.

### Commit 3b: move Python tree + docs into `_legacy/`
```
chore(legacy): move python source, tests, and docs to _legacy/
```

**Move into `_legacy/` (use `git mv` so rename detection works):**

| Source | Destination |
| --- | --- |
| `audio_transcribe/` | `_legacy/audio_transcribe/` |
| `transcribe.py` | `_legacy/transcribe.py` |
| `build.py` | `_legacy/build.py` |
| `test_all_apis.py` | `_legacy/test_all_apis.py` |
| `analyze_jsons.py` | `_legacy/analyze_jsons.py` |
| `batch_templates/` | `_legacy/batch_templates/` |
| `pyproject.toml` | `_legacy/pyproject.toml` |
| `uv.lock` | `_legacy/uv.lock` |
| `.python-version` | `_legacy/.python-version` |
| `tests/` *(Python files only)* | `_legacy/tests/` |
| `docs/` *(entire tree)* | `_legacy/docs/` |

**Stays at root (excluded from the move):**
- `cmd/`, `internal/`, `go.mod`, `go.sum`
- `.claude/`, `.git/`, `.gitignore` (update to drop Python ignores in commit 3d)
- `CLAUDE.md` (will be rewritten for Go after merge; out of scope here)
- `README.md`, `CHANGELOG.md` (Python-era content; flagged for rewrite, but left in place this round — moving them risks dangling references that confuse `git log --follow` later)
- `ROADMAP.md` (just created in 3a)
- Any `*_test.go` files inside `tests/` (Go tests stay)

Specifically for `tests/`: move every non-Go file into `_legacy/tests/`. If a `tests/integration/cli_test.go` (or similar) exists, keep it at `tests/integration/cli_test.go`. After the move, `tests/` may be nearly empty — that's fine, it becomes the Go integration test home.

### Commit 3c: delete `_legacy/`
```
chore(legacy): remove archived python tree
```
- `git rm -r _legacy/`
- No other changes.

### Commit 3d (optional, if needed): tidy `.gitignore`
```
chore: drop python-only gitignore entries
```
- Remove `__pycache__/`, `*.pyc`, `.venv/`, `uv.lock` (if listed), etc. from `.gitignore`.
- Only if removing those leaves the file logically cleaner; otherwise fold into 3c.

## Stage 4 — Merge to `main`

```bash
git checkout main
git pull
git merge worktree-go-port      # fast-forward if main hasn't moved
                                 # else falls back to a merge commit
git push
```

After successful merge:
- `git worktree remove .claude/worktrees/go-port` (or use the harness's worktree exit flow)
- `git branch -d worktree-go-port` (only after confirming the merge landed on origin)

## Critical files referenced

**Plan output files (new):**
- `docs/plans/2-feature-parity-completion.md` (Stage 1)
- `ROADMAP.md` at repo root (Stage 3a, copy of above)

**Go feature files Stage 1 will eventually touch (not modified in *this* plan):**
- `internal/core/domain/transcription.go` (extend `DaVinciOptions`, add `FormatWordSRT`)
- `internal/core/domain/ids.go` (add `FormatWordSRT`)
- `internal/core/services/davinci.go` (`applyDavinci` — wire padding)
- `internal/adapters/format/srt.go`, `davinci.go` (char-length wrapping, word-srt)
- `internal/adapters/api/elevenlabs/client.go:140` (un-hardcode diarize)
- `internal/adapters/api/assemblyai/client.go` (speaker_labels)
- `internal/delivery/cli/transcribe.go` (new flags)
- `internal/delivery/tui/app.go`, `options.go` (interactive mode)

**Existing helpers Stage 1 should reuse:**
- `services/davinci.go:applyDavinci` for all DaVinci-time transforms (padding, filler uppercasing, pause markers)
- `services/chunking.go:mergeChunks` for any word-stream aggregation
- `adapters/format/grouping.go` for SRT block grouping logic

## Verification

After all stages land on `worktree-go-port`:

1. **Build:** `go build ./...` succeeds.
2. **Tests compile:** `go vet ./...` clean.
3. **Tests run:** `go test ./...` — all pass or skip; **zero failures**. Count of skipped tests is the visible parity-gap meter.
4. **Repo cleanliness:** `git status` clean; `ls` at repo root shows only Go artefacts (`cmd/`, `internal/`, `tests/`, `go.mod`, `go.sum`, `ROADMAP.md`, `CLAUDE.md`, `README.md`, `CHANGELOG.md`, `.claude/`, `.gitignore`, `.git/`). No `audio_transcribe/`, no `_legacy/`, no `pyproject.toml`.
5. **History:** `git log --oneline worktree-go-port ^main` shows the three (or four) cleanup commits with clear messages.
6. **Roadmap visible:** `cat ROADMAP.md` shows the feature parity plan content.
7. **Post-merge:** `git checkout main && git log --oneline -5` shows the cleanup commits at the tip.
