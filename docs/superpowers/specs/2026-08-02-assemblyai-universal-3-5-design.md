# AssemblyAI: auto model selection and Universal-3.5 Pro migration

**Date:** 2026-08-02
**Status:** Approved design, ready for implementation planning

## Problem

AssemblyAI deprecates `universal-3-pro` on **2026-09-02**. After that date any request
that explicitly names it returns an error. Our AssemblyAI adapter hardcodes
`universal-3-pro` as the default model and always sends an explicit `speech_models`
array, so every job would fail.

AssemblyAI's guidance is to either pin the replacement (`universal-3-5-pro`) or omit
`speech_models` entirely and let the API pick the current flagship. Omitting is the
better default: per the
[model selection docs](https://www.assemblyai.com/docs/pre-recorded-audio/select-the-speech-model),
an omitted `speech_models` behaves as `["universal-3-5-pro", "universal-2"]` with
automatic fallback to Universal-2 for languages outside Universal-3.5 Pro's 18. That is
exactly what our hand-rolled fallback array reimplements — with the difference that
AssemblyAI updates their list and we would have to update ours on every release.

## Goals

- Default AssemblyAI jobs to omitting `speech_models`, so the API always picks its
  current best model.
- Keep explicit pinning available for users who need a specific model.
- Migrate saved configs, batch templates, and scripts that still say `universal-3-pro`
  without breaking them, while telling the user their input is stale.
- No behavior change for any other provider.

## Non-goals

- Changing the `--speech-models` flag's shape or the `ports.ProviderOpts.SpeechModels`
  contract.
- Touching other providers, or the legacy Python package under `build/`.
- Model discovery for AssemblyAI (they have no discovery endpoint; the list stays
  hardcoded).

## Decisions

Two questions were settled up front:

**Explicit pins still get the Universal-2 safety net.** When a user names a model, the
request is `[chosen..., universal-2]`, as today. A pinned job never fails purely because
the language falls outside the pinned model's coverage.

**Deprecated IDs are remapped, with a warning.** `universal-3-pro` in any input is
rewritten to `universal-3-5-pro` and a one-line deprecation notice is logged. Old
configs keep working after 2026-09-02 without user action, but the user learns their
config is stale.

## Design

### 1. Model catalog — `internal/adapters/api/assemblyai/models.go`

```go
const (
    AutoModel     = "auto"        // sentinel: omit speech_models entirely
    fallbackModel = "universal-2"
)

// deprecatedModels maps retired IDs to their replacement.
// AssemblyAI retires universal-3-pro on 2026-09-02.
var deprecatedModels = map[string]string{"universal-3-pro": "universal-3-5-pro"}
```

- `Models()` returns `["auto", "universal-3-5-pro", "universal-2"]`. `auto` is first so
  `pickDefaultModel` (`gui/mainwindow.go:482`) and every dropdown land on it even when a
  curated remote list is in play.
- `DefaultModel()` returns `AutoModel`. The pipeline's existing
  `model == "" → prov.DefaultModel()` substitution (`core/services/pipeline.go:67`) then
  yields `auto` for unset input, with no core change.
- `modelCaps` gains entries for `auto` and `universal-3-5-pro` and drops
  `universal-3-pro`. All three use the existing `sttCaps` (word timestamps, diarization,
  language hint). Capabilities for `auto` are the union of what any auto-selected model
  can do; the family is uniform on these axes, so nothing is lost. `Capabilities` keeps
  returning the zero value for unknown IDs — fail-safe.
- The file-header comment explaining why only two models are advertised is rewritten for
  the new set.

### 2. Normalization — pure function in `models.go`

```go
// NormalizeModels rewrites deprecated model IDs, preserving order. It returns the
// cleaned list and the original IDs that were remapped, for a deprecation warning.
func NormalizeModels(in []string) (out, remapped []string)
```

Pure and idempotent, so it is safe to call from the client and from any future
input-validation site without double-remapping. Unknown IDs pass through untouched:
AssemblyAI's 400 response already names the valid set, and silently rewriting an
unrecognized string would hide real typos.

### 3. Request building — `internal/adapters/api/assemblyai/client.go`

`buildSpeechModels(primary string, fallbacks []string) []string` keeps its signature and
gains one branch:

1. Normalize `primary` and every entry of `fallbacks` via `NormalizeModels`.
2. If `fallbacks` is empty **and** `primary` is `auto` or `""` → return `nil`.
3. Otherwise, as today: start from `fallbacks` when non-empty, else `[primary]`; append
   `fallbackModel`; dedupe preserving order, skipping empties. An `auto` primary with
   explicit fallbacks uses the fallbacks — the explicit pin wins.

`submitTranscript` sets `body["speech_models"]` only when the returned slice is
non-empty. In auto mode the key is absent from the JSON body entirely. `audio_url`,
`language_code` / `language_detection`, `speaker_labels`, `speakers_expected`, and
`keyterms_prompt` are unchanged.

### 4. Deprecation warning

`Client` gains a nil-safe chained setter:

```go
func (c *Client) WithLogger(l ports.Logger) *Client { c.log = l; return c }
```

`internal/delivery/wire.go:31` becomes
`assemblyai.New(k, httpClient).WithLogger(log)`. `internal/delivery/tui/wizard.go:43`'s
`assemblyai.New(key, nil)` keeps compiling unchanged (nil logger → no warning, and the
remap still happens).

When `NormalizeModels` reports a remap, `submitTranscript` logs once per request:

> `universal-3-pro is deprecated (AssemblyAI retires it 2026-09-02); using universal-3-5-pro. Update your config or --speech-models.`

Placing the warning in the adapter means one implementation covers the CLI, GUI, TUI,
and batch templates, instead of three copies at the input edges that would drift.

### 5. Result reporting — `parse.go`

In auto mode `parse` would stamp `Model: "auto"`, which tells the user nothing about
what actually ran. AssemblyAI echoes the model used in the response body, so:

- `transcriptResponse` gains a `SpeechModel` field bound to the `speech_model` JSON key.
- `parse` records `resp.SpeechModel` as `Result.Model` when the requested model was
  `auto` and the field is non-empty; otherwise it records the requested model as today.

The sidecar JSON then names the real model. Note that older AssemblyAI responses report
legacy labels here (the `sample_long_audio_assemblyai.json` fixture shows
`"speech_model": "best"`); we record whatever the API returns rather than interpreting
it.

### 6. Tests

Table-driven on `buildSpeechModels`:

| Input | Expected |
|---|---|
| primary `auto`, no fallbacks | `nil` |
| primary `""`, no fallbacks | `nil` |
| primary `auto`, fallbacks `["universal-3-5-pro"]` | `["universal-3-5-pro", "universal-2"]` |
| primary `universal-3-pro`, no fallbacks | `["universal-3-5-pro", "universal-2"]` |
| primary `universal-2`, no fallbacks | `["universal-2"]` (no duplicate) |
| fallbacks `["universal-3-pro", "universal-2"]` | `["universal-3-5-pro", "universal-2"]` |

Client-level: a test asserting the submitted body has **no** `speech_models` key in auto
mode. The existing `TestAssemblyAI_SpeechModelsDefaultsToFallback`
(`client_test.go:218`) inverts into this. `TestAssemblyAI_RequestIncludesSpeechModels`
(`client_test.go:208`) currently pins `universal-3-pro` and must be updated to the new
IDs.

`models_test.go`: `DefaultModel() == "auto"`; `Models()[0] == "auto"`; `auto` has word
timestamps (guards against the GUI greying out the SRT checkboxes for the default
selection); `universal-3-pro` is no longer advertised.

`NormalizeModels`: remaps the deprecated ID, reports it in `remapped`, passes unknown
IDs through, and is idempotent.

`parse`: a fixture with `speech_model` set records that value when the request was auto.

### 7. Docs and user-facing strings

- `README.md`: replace `universal-3-pro` references; document `auto` as the default and
  what it means (AssemblyAI picks its current best model).
- `internal/delivery/gui/mainwindow.go:196`: the speech-models placeholder currently
  reads `e.g. universal-3-pro,universal-2 …`; update to the new IDs.
- `CHANGELOG.md`: a `Changed` entry covering the new default, the remap, and the
  deprecation date.

## Risks

- **AssemblyAI adds a model whose capabilities differ from `sttCaps`.** Then `auto`'s
  advertised capabilities would over-promise. Low likelihood on the word-timestamp and
  diarization axes, and the server-side check remains the final gate.
- **The `speech_model` echo uses labels not in our catalog** (e.g. `best`). Accepted: we
  record it verbatim as an informational field, and never feed it back into a request.

## Verification

`go test ./...` green, plus a manual auto-mode run against the live API confirming the
transcript succeeds and the result's model field names a real model rather than `auto`.
