# Diarization & Dual-File Podcast Merge (Go port)

## Context

We want speaker diarization in the Go version of Transcribe, focused on ElevenLabs
Scribe. Two distinct needs:

1. **Single-file diarization** — one mixed recording, provider returns anonymous
   speakers (`speaker_0`, `speaker_1`, …). The golden fixtures the user produced from
   the ElevenLabs web UI live at `tests/fixtures/audio_files/diarization_sample.mp3*`
   and number speakers from zero.
2. **Dual-file podcast merge** — our podcasts record each participant on a separate
   track (`CON-XXX_julia.wav`, `CON-XXX_guest.wav`). Transcribe each separately (each is
   effectively single-speaker), then interleave into one transcript with **user-supplied
   labels** ("Julia", "Gast"). This is more reliable than API diarization and is the
   primary production workflow.

**On "saved speakers":** ElevenLabs Scribe STT has no enrolled/named-speaker feature —
diarization is anonymous per request. Auto-detecting "Julia" via the API is not
possible; the dual-mic approach (Scheme 2) is the correct substitute.

Current state (already implemented in Go): the ElevenLabs client sends `diarize` +
`speakers_expected` (`internal/adapters/api/elevenlabs/client.go`), the parser fills
`Word.Speaker` + `Result.Speakers` (`.../elevenlabs/parse.go`), CLI flags `--diarize`,
`--speaker-labels`, `--num-speakers` exist (`internal/delivery/cli/transcribe.go`), and
the SRT writer prefixes `[Speaker X]:` (`internal/adapters/format/srt.go`). So Scheme 1
is mostly done; the gaps are ID normalization and text-format labels.

## Scope (decided)

- **Scheme 1 polish:** normalize raw speaker IDs (`speaker_0` → `0`) and add speaker
  labels to plain-text output. **Skip** the golden SRT dialogue-dash styling for now.
- **Scheme 2:** new `merge` subcommand; per-file user-supplied labels; support an
  optional per-file time offset (tracks may not share an exact zero point).

---

## Part A — Scheme 1: single-file diarization polish

### A1. Normalize speaker IDs

Provider responses vary: ElevenLabs returns `speaker_0`, AssemblyAI returns `A`/`B`.
Today the SRT writer emits `[Speaker speaker_0]:` (ugly). Normalize at the parse
boundary so `Word.Speaker` holds a clean token (`0`, `1`, `A`).

- Add a small helper `normalizeSpeakerID(string) string` that strips a case-insensitive
  `speaker_`/`speaker ` prefix and trims spaces; leave already-clean IDs (`A`) untouched.
- Apply it in `internal/adapters/api/elevenlabs/parse.go` where `w.SpeakerID` is read
  (both on `Word.Speaker` and the de-duped `Speakers` slice). Mirror in
  `internal/adapters/api/assemblyai/parse.go` for consistency (no-op for `A`/`B`).
- The SRT writer's existing `"[Speaker " + speaker + "]: "` then renders `[Speaker 0]:`,
  matching the golden output's `[Speaker 0]` numbering.

### A2. Speaker labels in plain-text output

`internal/adapters/format/text.go` ignores speakers. Add label emission gated on
`WriteOpts.SpeakerLabels`: when the speaker changes between consecutive words, start a
new line/paragraph prefixed with `[Speaker X]: `. Match the grouping the text formatter
already uses; keep behavior identical when `SpeakerLabels` is false.

### A3. Validate against the golden fixture

The user's golden `.json` is the **web-UI export shape** (`segments[]` with
`speaker:{id,name}`), which differs from the **API response shape** the parser targets
(`words[]` with `speaker_id`). Do **not** retarget the parser to the export shape.
Instead:

- Capture one real ElevenLabs API response for `diarization_sample.mp3` (run the live
  endpoint once, or reuse a cached sidecar) and store it as a parser test fixture.
- Add a parse test asserting normalized IDs (`0`, `1`) and that speaker turns appear.
- Add/confirm an SRT speaker test produces `[Speaker 0]:` / `[Speaker 1]:` prefixes.

### Files (Part A)
- `internal/adapters/api/elevenlabs/parse.go` (+ new `normalizeSpeakerID`)
- `internal/adapters/api/assemblyai/parse.go`
- `internal/adapters/format/text.go` + `text_test.go`
- `internal/adapters/format/srt_speaker_test.go` (extend)
- `internal/adapters/api/elevenlabs/parse_test.go` + new testdata fixture

---

## Part B — Scheme 2: dual-file podcast merge

### B1. CLI surface — new `merge` subcommand

```
transcribe merge --speaker Julia=CON-259_julia.wav \
                 --speaker Gast=CON-259_guest.wav \
                 [--offset Gast=1.2s] \
                 --api elevenlabs [--language de] [--output srt,text,json]
```

- `--speaker LABEL=PATH` (repeatable, 2+ allowed) — label is the user-supplied name.
- `--offset LABEL=DURATION` (repeatable, optional) — per-track time shift to align
  tracks that don't share an exact zero point (default 0).
- Reuse existing transcribe flags (`--api`, `--model`, `--language`, `--output`,
  output-dir, cache, char/word wrapping). New command file:
  `internal/delivery/cli/merge.go`, registered alongside `transcribe.go`.

### B2. Transcribe each track, then interleave

Reuse the existing pipeline via `services.Service.Submit(ctx, req)`
(`internal/core/services/service.go:106`) — call it once per input file. Each track is
single-speaker, so **do not** request provider diarization; instead overwrite every
`Word.Speaker` in that track's `Result` with the user's label.

New domain logic (pure, unit-testable), e.g. `internal/core/services/merge.go`:

- `MergeResults(tracks []LabeledResult) *domain.Result` where `LabeledResult` carries
  the `*domain.Result`, its label, and its offset.
- Apply each track's offset to every word's `Start`/`End`, set `Word.Speaker = label`,
  concatenate all words, then **stable-sort by Start**. Rebuild `Result.Text` and
  `Result.Speakers` (one entry per label, `Label` set to the user name).

### B3. Output

Three sets of outputs are produced:

1. **Per-track (Julia):** SRT + JSON + text, written from that track's individual
   `Result` (single-speaker; standard formatters, no label prefix needed).
2. **Per-track (Gast):** SRT + JSON + text, same.
3. **Combined:** SRT + text **only** — no combined JSON. The merged transcript is always
   derived from the two separated JSON sidecars, so a combined JSON would be redundant.

Write each track's individual `Result` (per the requested `--output` formats) first, then
build the merged `Result` and emit only the combined SRT + text. Feed the merged result
through the existing writers with `WriteOpts.SpeakerLabels = true`. Because labels are
full names, the SRT/text prefix becomes `[Julia]:` / `[Gast]:` rather than `[Speaker X]:`
— adjust the writer prefix so a named (non-bare-diarization-token) speaker renders as
`[<label>]:`, skipping the literal word "Speaker".

Naming: per-track files use each track's base name (e.g. `CON-259_julia.srt`); the
combined files get a distinct suffix (e.g. `CON-259_combined.srt` / `.txt`).

### Files (Part B)
- `internal/delivery/cli/merge.go` (new) + flag parsing tests
- `internal/core/services/merge.go` (new) + `merge_test.go`
- `internal/adapters/format/srt.go` + `text.go` (named-label prefix rendering)
- Register subcommand in `cmd/transcribe/main.go`

---

## Part C — TUI / GUI (assessment only; not built now)

CLI-first per the user. Rough effort estimate to revisit later:

- **Scheme 1 (single-file):** trivial — the TUI/GUI already surface speaker options
  (`internal/delivery/tui/options.go`, `internal/delivery/gui/mainwindow.go` reference
  Speaker). Mostly already covered once Part A lands.
- **Scheme 2 (merge):** moderate — both UIs assume a single input (or a folder). A merge
  flow needs a two-file picker with a label field per file (and optional offset). In the
  GUI that's a new dialog/mode; in the TUI an added wizard branch. Defer until the CLI
  workflow is proven in production.

---

## Verification

1. **Unit/parser:** `go test ./internal/adapters/api/elevenlabs/... ./internal/adapters/format/... ./internal/core/services/...`
   — covers ID normalization, text labels, and `MergeResults` interleaving/offset.
2. **Scheme 1 end-to-end:** run live against the golden sample and diff structure:
   ```
   go run ./cmd/transcribe tests/fixtures/audio_files/diarization_sample.mp3 \
     --api elevenlabs --diarize --output srt,text,json --language de
   ```
   Confirm `[Speaker 0]` / `[Speaker 1]` labels and turn boundaries roughly match the
   golden `.srt`/`.txt` (exact dialogue-dash styling is out of scope).
3. **Scheme 2 end-to-end:** run the real podcast pair and inspect the merged transcript:
   ```
   go run ./cmd/transcribe merge \
     --speaker Julia="G:\...\CON-259 - Jasmin Fuchs\CON-259_julia.wav" \
     --speaker Gast="G:\...\CON-259 - Jasmin Fuchs\CON-259_guest.wav" \
     --api elevenlabs --language de --output srt,text,json
   ```
   Verify: per-track Julia + Gast SRT/JSON/text sidecars exist; a combined SRT + text
   (no combined JSON) interleaves `[Julia]:` / `[Gast]:` turns in chronological order;
   and offsets shift a track as expected.
4. Update `CHANGELOG.md`; keep commits atomic (`feat(elevenlabs): normalize speaker ids`,
   `feat(merge): dual-track podcast merge`, etc.).
