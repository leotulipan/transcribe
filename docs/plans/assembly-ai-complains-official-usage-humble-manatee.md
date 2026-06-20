# Capability-aware UI + AssemblyAI/OpenAI fixes

## Context

Several issues surfaced after the GUI/Windows polish work:

1. **AssemblyAI rejects requests.** We send `speech_model` (singular) on every
   request and *additionally* send `speech_models` (array) only when a fallback
   list is set. Official usage is the plural array; the singular field is the
   likely cause of the complaint.
2. **OpenAI lists every model** (chat, embeddings, TTS…) in the picker, not just
   speech-to-text models.
3. **The GUI shows options that don't apply** to the selected provider — the
   "Provider hints (assemblyai)" and "Diarization" sections are always visible,
   and the SRT/word_srt/davinci_srt checkboxes are always enabled even for
   text-only providers (Gemini, Mistral).
4. **On Start, the progress area isn't in view** — the user can't see progress
   without manually scrolling.

A per-model capability model **already exists and is enforced server-side**
(`ports.Provider.Capabilities(model) ModelCapabilities`, `OutputFormat.NeedsTimestamps()`,
and `checkCapabilities` in `services/pipeline.go`). The UI simply can't read it:
`ports.TranscribeService` exposes `ListModels`/`DefaultModel` but not capabilities.
So most of this work is **surfacing existing truth**, plus two small API fixes and
one UX tweak.

### Decisions (confirmed with user)
- **AssemblyAI:** always send `speech_models` as an array, drop the singular
  `speech_model`. The array is the selected model followed by `universal-2` as a
  resilience fallback — `universal-2` is appended to **every** request (the
  GUI/TUI have no fallback-selection control). An explicit CLI fallback list is
  honored, with `universal-2` still ensured as the final entry.
- **Gating scope:** wire capability gating into **both** the GUI and the TUI.

### Capability reference (from `internal/adapters/api/*/models.go`)
| Provider | WordTimestamps (SRT ok?) | Diarization |
|---|---|---|
| assemblyai | yes | yes |
| elevenlabs | yes | yes |
| openai | yes | no |
| groq | yes | no |
| gemini | **no** (text only) | no |
| mistral | **no** (text only) | no |

---

## Changes

### A. Shared capability access (enabler)
- `internal/ports/service.go` — add to `TranscribeService`:
  `Capabilities(p domain.ProviderID, model string) (ModelCapabilities, bool)`
  (`ok=false` when the provider isn't wired up).
- `internal/core/services/service.go` — implement it next to `ListModels`
  (~line 77), using `providerFor(s.deps, id)` / `s.deps.Providers[id]` then
  `p.Capabilities(model)`. This single method feeds both UIs.

### B. AssemblyAI `speech_models` fix
- `internal/adapters/api/assemblyai/client.go`, `submitTranscript` (lines 176–198):
  - Remove `"speech_model": model` (line 180).
  - Build the `speech_models` array via a small helper: start from
    `opts.SpeechModels` if non-empty, else `[]string{model}`; then ensure
    `universal-2` is the final entry (append if absent); dedupe preserving order.
    Set `body["speech_models"]` to the result. Fold away the conditional at 196–198.
  - Add `const fallbackModel = "universal-2"` in the assemblyai package.
- `internal/adapters/api/assemblyai/client_test.go`: existing
  `TestAssemblyAI_RequestIncludesSpeechModels` still passes (its explicit list
  now gets `universal-2` ensured as last — update its expectation). Add cases:
  - empty `SpeechModels`, model `universal-3-pro` → `["universal-3-pro","universal-2"]`, no `speech_model` key;
  - model `universal-2` → `["universal-2"]` (no duplicate);
  - explicit `["slam-1"]` → `["slam-1","universal-2"]`.
- The GUI "Speech models (csv)" field remains an optional override; it just feeds
  `opts.SpeechModels`.
- Verify the param against AssemblyAI docs/live during implementation (this is
  the change that triggered the original complaint).

### C. OpenAI STT-only model filtering
- `internal/adapters/api/openai/discover.go`, `DiscoverModels` (lines 46–50):
  filter ids to STT models — keep those whose id contains `whisper` or
  `transcribe` (covers `whisper-1`, `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`,
  `gpt-4o-transcribe-diarize`, and future names). If the filtered list is empty,
  fall back to `openai.Models()`. Mirrors the ElevenLabs pattern
  (`elevenlabs/discover.go` lines 19, 57–60). Update the comment at lines 14–16.
- `internal/adapters/api/openai/discover_test.go`: assert chat/embedding ids are
  dropped and STT ids kept.

### D. GUI: scroll to progress on Start
- `internal/delivery/gui/mainwindow.go`: store the main scroll container on the
  struct (`formScroll *container.Scroll`; assign `m.formScroll = scrolled` at
  line 305). In `onStart` (~line 443), after launching the job, call
  `m.formScroll.ScrollToBottom()` so the Progress section + buttons come into view.

### E. GUI: provider-aware sections + greyed formats
- In `newMainWindow` (lines 220–273), build the two provider-specific accordion
  items as named vars stored on the struct: `m.advanced *widget.Accordion`,
  `m.diarItem`, `m.hintsItem *widget.AccordionItem`. Keep the always-present items
  in a base slice.
- Add `applyCaps(provider, model string)` on `mainWindow`:
  - `caps, ok := m.deps.Service().Capabilities(...)`; if `!ok` leave everything on.
  - Rebuild `m.advanced.Items`: include `m.diarItem` only if `caps.Diarization`;
    include `m.hintsItem` only if `provider == "assemblyai"`; then
    `m.advanced.Refresh()`.
  - Formats: if `!caps.WordTimestamps`, uncheck + `Disable()` `m.fmtSRT`,
    `m.fmtWordSRT`, `m.fmtDavinci` and ensure `m.fmtText` is checked; else
    `Enable()` them.
- Call `applyCaps` from `onProviderChanged` (lines 367–384) after the model is
  selected, **and** from a new `m.model.OnChanged` (caps are per-model for Groq).
- `lockUI` (lines 778–809): after the unlock branch re-enables controls, call
  `applyCaps(...)` so capability-disabled format boxes aren't wrongly re-enabled.

### F. TUI: same gating
- `internal/delivery/tui/options.go`: provider+model are known by `stepFormats`.
  - `buildFormatList` (and its call site ~line 88): when
    `!caps.WordTimestamps`, offer only `text` (omit srt/word_srt/davinci_srt).
  - Advanced screen: omit the `advFieldDiarize` row when `!caps.Diarization`
    (adjust the `advancedField` iteration/render so the index set is built from
    caps rather than the fixed enum).
  - Drop any prefilled-but-unsupported format (from CLI flags) with a brief notice
    instead of silently submitting an invalid combo.

---

## Verification
- Build/test with CGO: `pwsh scripts/build.ps1` (the PowerShell tool lacks gcc on
  PATH; build via the script or rely on CI). Then `go test ./...`.
- Unit: assemblyai `client_test` (plural array present, singular absent), openai
  `discover_test` (filtering), a services test for `Capabilities`.
- Manual GUI (`transcribe --ui=gui`): switch providers —
  - Gemini/Mistral → srt/word_srt/davinci greyed, only `text` selectable;
  - Diarization section hidden for openai/groq/gemini/mistral, shown for
    assemblyai/elevenlabs; Provider hints shown only for assemblyai;
  - click Start → form scrolls to the Progress section.
- Manual TUI (`transcribe --ui=tui`): pick Gemini → formats list offers only
  `text`; advanced screen omits the diarize row.
- Live AssemblyAI run with and without `--speech-models "universal-3-pro,universal-2"`
  → both return 2xx.

## Out of scope
- Code signing / notarization.
- Promoting the hardcoded "Provider hints" text into structured metadata (keep the
  current hints; only gate visibility).
- Capability gating for any options beyond formats and diarization.
