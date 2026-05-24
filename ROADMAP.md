# Feature Parity Completion (Python → Go)

> Snapshotted from `docs/plans/2-feature-parity-completion.md` on 2026-05-18 before the Python retirement. The source plan is gone after `_legacy/` deletion; this file is the surviving roadmap on `main`.

## Context

Six Python-era features survive only in the Python tree (or as half-wired Go scaffolding) at the point the Go port branch merges to `main`. This plan is the punch list to close them out after the merge. Each phase produces shippable CLI changes; phases are ordered from "easiest, highest-leverage" to "largest scope," so we can ship value as soon as Phase 1 lands rather than waiting for the TUI rebuild at the end.

The matching test scaffolds are already in the Go tree as `t.Skip()` stubs (see Stage 2 of `1-do-we-have-shiny-summit.md`). Each phase below converts its skipped tests into real assertions — that's the Red → Green signal.

## Phase 1 — Wire existing logic to CLI (Easy)

The Go pipeline already implements pause markers and filler uppercasing inside `internal/core/services/davinci.go:applyDavinci`. The padding-start field exists on `DaVinciOptions` but is never read. Phase 1 is plumbing only — no new logic.

### 1a. Filler-words flags
- **Add flags** in `internal/delivery/cli/transcribe.go`:
  - `--filler-words` (comma-separated, defaults to `DefaultFillerWords`)
  - `--remove-fillers` (strip matches from `r.Words` entirely)
  - `--filler-lines` (current uppercase behavior; default true to keep parity with current Go output)
- **Wire to** `domain.DaVinciOptions.FillerWords` on the `Request` build at `transcribe.go:101-105`.
- **Add a `RemoveFillers bool`** field to `DaVinciOptions` (`internal/core/domain/transcription.go:48`). `applyDavinci` checks it and skips appending the word entirely when set.

### 1b. Padding-start logic
- **Implement in** `internal/core/services/davinci.go:applyDavinci`: after the existing pause-marker loop, walk `out` and for each word `w` subtract `min(opts.PaddingStart, gap_before_w / 2)` from `w.Start`. Matches the Python `apply_intelligent_padding` heuristic at `audio_transcribe/transcribe_helpers/output_formatters.py:303`.
- **Add flag** `--padding-start` (ms, default 0) in `transcribe.go`.
- **Wire to** `DaVinciOptions.PaddingStart` on the request build.

### Verification (Phase 1)
- `go test ./internal/core/services -run TestDavinci` — `davinci_padding_test.go` (currently skipped) flips green.
- `go run ./cmd/transcribe --help` shows the three new filler flags + `--padding-start`.
- Manual: run with `--davinci --padding-start 80 --filler-words um,uh,ähm` on a sample with known pauses; inspect the `.srt` for shifted starts.

## Phase 2 — Output format extensions (Medium)

### 2a. Word-level SRT
- **Add** `FormatWordSRT OutputFormat = "word_srt"` in `internal/core/domain/ids.go:20`.
- **Implement** `internal/adapters/format/word_srt.go` mirroring `srt.go` but emitting one subtitle per word (use `r.Words` directly, skip the `groupWords` call). Reference: Python `audio_transcribe/utils/formatters.py` `create_srt` "word" mode.
- **Register** in the format adapter wiring (the place `NewSRT()` is plugged in — check `internal/delivery/cli/deps.go` or wherever the format port is bound).
- **Add CLI flag** `--word-srt` (boolean convenience flag, like `--davinci`) and route through `parseFormats` in `transcribe.go:113-134`.

### 2b. `--chars-per-line` SRT wrapping
- **Add** a `MaxCharsPerLine int` config to a new options struct (`SRTOptions`) or extend `DaVinciOptions` (preferred — DaVinci already wraps long blocks).
- **Implement wrapping helper** in `internal/adapters/format/grouping.go` (next to `groupWords`): split a block's word run into lines so no line exceeds N chars (greedy fill). Reference: Python `wrap_text_for_srt()` at `audio_transcribe/transcribe_helpers/output_formatters.py:33`.
- **Use the helper** in `srt.go:Write` and `davinci.go:Write` when `MaxCharsPerLine > 0`.
- **Add CLI flag** `--chars-per-line` (int, 0 = off; Python default was 55). Wire through `Request`.

### Verification (Phase 2)
- `go test ./internal/adapters/format -run TestWordSRT` — `word_srt_test.go` flips green.
- `go test ./internal/adapters/format -run TestSRTCharLen` — `srt_charlen_test.go` flips green.
- Update golden files under `internal/adapters/format/testdata/` if existing tests need new expected output.
- Manual: `transcribe sample.mp3 --output srt --chars-per-line 40` — eyeball the wrapping.

## Phase 3 — Provider features (Medium)

### 3a. Diarization plumbing
- **Add** `SpeakerLabels bool` to `domain.Request` (or to a new `domain.ProviderOptions` substruct if we want to keep `Request` lean).
- **ElevenLabs**: replace the hardcoded `diarize=false` at `internal/adapters/api/elevenlabs/client.go:140` with `req.SpeakerLabels`. Parse the speaker info in the response parser → populate `Result.Speakers` and `Word.SpeakerID`-equivalent (add a `Speaker` field on `domain.Word` if not there).
- **AssemblyAI**: add `speaker_labels=true` to the request payload at `internal/adapters/api/assemblyai/client.go`. Parse speakers same as above.
- **Other providers**: return a clear error or silently ignore (document in the plan: Groq/OpenAI/Gemini/Mistral don't expose diarization).

### 3b. Speaker labels in SRT output
- In `internal/adapters/format/srt.go` (and `davinci.go`), if any `Word.Speaker` is non-empty, prefix each block with `[SpeakerLabel]:` (or whatever Python emits — check `audio_transcribe/transcribe_helpers/output_formatters.py`).
- Gate behind `--speaker-labels` flag (default false, on by default if `--diarize` was set).

### 3c. CLI flags
- **Add** `--diarize` (boolean, default false) and `--speaker-labels` (boolean; if unset, mirrors `--diarize`).
- Wire to `Request` in `internal/delivery/cli/transcribe.go`.

### Verification (Phase 3)
- `go test ./internal/adapters/api/elevenlabs -run TestDiarize` — `diarize_test.go` flips green.
- Manual: `transcribe sample-multi-speaker.wav --api elevenlabs --diarize --output srt` — inspect speaker prefixes in the .srt.

## Phase 4 — TUI completion (Hard)

Today `internal/delivery/tui/app.go` has three screens: file picker → options → progress. The Python TUI (`audio_transcribe/tui/wizard.py` + `interactive.py`) is richer: interactive setup wizard for API keys, model selection from live API listing, language picker.

### 4a. Setup wizard
- **New screen** `internal/delivery/tui/wizard.go` triggered by `transcribe --setup`.
- For each provider: ask for API key, call `provider.CheckAPIKey()` (port the Python `check_api_key` pattern — already partly in `internal/adapters/api/*/client.go`), report green/red.
- Persist via `ports.Config.SetAPIKey(provider, key)` (extend the config port if it lacks this method).

### 4b. Interactive mode (run with no args)
- The existing `EscalateToTUI` path at `internal/delivery/cli/transcribe.go:36-46` already covers the "no files passed" case. Extend the TUI Options screen to also pick:
  - Model (driven by live `provider.ListModels()` — uses the already-built discovery interfaces from the earlier 4-plan roadmap)
  - Language (drop-down of common ISO-639-1)
  - Filler/diarization/padding options once Phases 1-3 land
- Keep the current Charmbracelet bubbletea architecture; add sub-screens by extending the `screen` enum at `internal/delivery/tui/app.go:30-40`.

### Verification (Phase 4)
- `go test ./internal/delivery/tui -run TestSetupWizard` — wizard flow covered with a fake `ports.Config`.
- `go test ./internal/delivery/tui -run TestE2E` (existing `e2e_test.go`) extended to cover the new option screens.
- Manual: `transcribe --setup` walks through every provider; `transcribe` (no args) opens the full picker.

## Critical files

**Domain (touch once, consume everywhere):**
- `internal/core/domain/transcription.go` — `DaVinciOptions{RemoveFillers}`, `Word{Speaker}`, `Request{SpeakerLabels}`
- `internal/core/domain/ids.go` — `FormatWordSRT`

**Pipeline core:**
- `internal/core/services/davinci.go:applyDavinci` — padding logic (Phase 1b), remove-fillers branch (Phase 1a)

**Adapters:**
- `internal/adapters/format/srt.go`, `davinci.go`, `word_srt.go` (new), `grouping.go` (wrap helper)
- `internal/adapters/api/elevenlabs/client.go:140` — un-hardcode diarize
- `internal/adapters/api/assemblyai/client.go` — add speaker_labels payload field

**Delivery:**
- `internal/delivery/cli/transcribe.go:50-60` — new CLI flags
- `internal/delivery/tui/wizard.go` (new) + extensions to `app.go`, `options.go`

## Existing helpers to reuse

- `internal/core/services/davinci.go:applyDavinci` — single point for all DaVinci transforms (do NOT build a parallel padding pipeline)
- `internal/adapters/format/grouping.go:groupWords` — pair with new `wrapByChars` for Phase 2b
- The discovery interfaces under `internal/ports/discovery/` (added in the earlier 4-plan roadmap) — feed model dropdowns in Phase 4b
- `internal/adapters/config/tomlstore.go` — extend for `SetAPIKey` in Phase 4a, don't roll a new config layer

## Test coverage map

Each phase removes `t.Skip(...)` from one or more files:

| Phase | Test file (skip → assertions) |
| --- | --- |
| 1a | `internal/core/services/davinci_text_test.go` (filler cases — partially already) |
| 1b | `internal/core/services/davinci_padding_test.go` |
| 2a | `internal/adapters/format/word_srt_test.go` |
| 2b | `internal/adapters/format/srt_charlen_test.go` |
| 3a | `internal/adapters/api/elevenlabs/diarize_test.go` |
| 4a | `internal/delivery/tui/wizard_test.go` (to be created in Phase 4) |

When the last skip flips, the Python feature gap is closed.

## Verification (end-to-end, after all four phases land)

- `go run ./cmd/transcribe --help` lists: `--word-srt`, `--chars-per-line`, `--padding-start`, `--filler-words`, `--remove-fillers`, `--filler-lines`, `--diarize`, `--speaker-labels`.
- `go test ./...` runs zero skips for the feature-parity test set.
- Golden fixtures in `internal/adapters/format/testdata/` regenerated and reviewed.
- Manual smoke: ElevenLabs run with `--diarize --davinci --padding-start 80 --chars-per-line 55 --filler-lines` produces a Resolve-ready .srt with speaker prefixes, intelligent padding, char-wrapped lines, and UPPERCASE filler lines.

---

## Phase 5 — Remaining Python CLI surface (added 2026-05-24)

Surfaced after sweeping the Python `cli.py` (preserved at `G:\Meine Ablage\_2_Areas\Scripts\Transcribe\build\lib\audio_transcribe\cli.py`) against the current Go flag set in `internal/delivery/cli/transcribe.go`. These were not in Phases 1–4. Group by area; each bullet is the Python flag, default, and where it must land in Go.

### 5a. Subtitle grouping / timing
- `--words-per-subtitle / -w` (int, default 0=off). Mutually exclusive with `--chars-per-line`. Pair with `groupWords` in `internal/adapters/format/grouping.go`.
- `--silent-portions / -p` (int ms, default per provider). Pause threshold; today hardcoded as `--silent-portion-ms 1500`. Rename or alias.
- `--padding-end` (int ms). Mirror of `--padding-start`; subtract from `Word.End` symmetrically in `applyDavinci`.
- `--show-pauses` (bool). Emit `(...)` text for pauses ≥ silent-portions. Already partially in `applyDavinci`; expose flag.
- `--start-hour` (int). Hour offset added to all SRT timestamps. Apply in `srt.go` / `davinci.go` write path.

### 5b. Frame-based editing (DaVinci)
- `--fps` (float), `--fps-offset-start` (int, default -1), `--fps-offset-end` (int, default 0). Snap word boundaries to frame grid before SRT emit. New helper next to `applyDavinci`.

### 5c. Provider tuning
- `--num-speakers` (1..32). AssemblyAI + ElevenLabs `speakers_expected` payload field. Gate on `--diarize`.
- `--keyterms-prompt` (string, AssemblyAI). Comma list. Pass through to AssemblyAI request.
- `--speech-models` (string, AssemblyAI). Fallback model list, e.g. `universal-3-pro,universal-2`. Pass through.

### 5d. Audio pipeline
- `--size-threshold` (float MB, default 100). Files under threshold skip conversion if format compatible. Wire in `internal/core/services/audio.go` (or equivalent).
- `--chunk-length` (int s, default 600), `--overlap` (int s, default 10). Chunking knobs already implied by the chunker; expose.
- `--use-input` (bool). Bypass conversion entirely, send original.
- `--use-pcm` (bool). Convert to PCM WAV instead of FLAC.
- `--keep-flac` (bool), `--keep` (bool). Retain intermediates instead of deleting.

### 5e. I/O & workflow
- `--force / -r` (bool). Re-transcribe even when sidecar transcript exists. Today `--use-cache=false` does this; add `-r` alias / rename.
- `--save-cleaned-json / -J` (bool). Persist the normalized pre-format JSON next to outputs.
- `--use-json-input / -j` (bool). Accept a previously-saved JSON as input and skip the API call.
- `--extensions / -e` (string). Filter file extensions when input is a directory.
- `--start-hour` — see 5a.

### 5f. Logging & discovery
- `--debug / -d`, `--verbose / -v`. Log level / console verbosity. Wire to the logger port.
- `--list` (bool). List APIs + their models and exit. Use the existing discovery interfaces.
- `--api-key` (string, with `--setup`). Non-interactive key setter for a single provider.

### 5g. Short-flag aliases for parity
The Python CLI exposes single-letter aliases (`-a --api`, `-l --language`, `-o --output`, `-c --chars-per-line`, `-C --word-srt`, `-D --davinci-srt`, `-m --model`, `-p --silent-portions`, `-w --words-per-subtitle`, `-d --debug`, `-v --verbose`, `-e --extensions`, `-r --force`, `-j --use-json-input`, `-J --save-cleaned-json`). Add via Cobra `Flags().StringVarP` etc. so muscle memory carries over.

### Verification (Phase 5)
- `go run ./cmd/transcribe --help` lists every Phase 5 flag.
- Run with `--use-json-input old.json --output srt` skips API entirely and re-emits formats from a cached JSON.
- Regression test: `--chunk-length 300 --overlap 5` produces multi-chunk output without word loss across boundaries.

---

## Phase 6 — Bug: stale `.transcribe_temp_*` directories (added 2026-05-24)

### Symptom
Empty `.transcribe_temp_<basename>` folders accumulate next to processed audio files and **nest recursively** on retries — observed under `tests/fixtures/audio_files/`, e.g. `.transcribe_temp_sample_video/.transcribe_temp_sample_video_intermediate_extracted/.transcribe_temp_sample_video_intermediate_extracted_intermediate_extracted/`. The cleanup pass that removed Python-era temp dirs (see Python `intermediate_files.py` cleanup) was not ported.

### Fix
- In whatever Go service creates the per-input temp dir (search `transcribe_temp` or `.transcribe_temp_`), wrap creation in a `defer` that removes the directory if empty after processing.
- On chunked / intermediate-extract retries, reuse the existing temp dir; do not derive a new name by appending `_intermediate_extracted` again. Idempotent naming = `<basename>` only, not whatever the prior dir name was.
- Add a startup sweep (gated behind `--clean` or run unconditionally if cheap) that removes empty `.transcribe_temp_*` dirs under the working directory.

### Verification
- `go test ./...` followed by `find tests/fixtures -type d -name '.transcribe_temp_*'` returns nothing.
- Manual: run twice in a row on the same input; only one `.transcribe_temp_<name>` exists during run, none after.

