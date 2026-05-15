# Go Port — Plan 2: Provider Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the five remaining providers (AssemblyAI, ElevenLabs, OpenAI, Gemini, Mistral) on top of the foundation Plan 1 produced. Each provider plugs into `ports.Provider` and gets wired into `delivery/wire.go` so the CLI's `--api` flag accepts it.

**Architecture:** Each provider lives in its own subpackage under `internal/adapters/api/<name>/` with the same shape as the Groq adapter from Plan 1: `models.go`, `parse.go`, `client.go`, and a build-tagged `integration_test.go`. Existing hexagonal boundaries unchanged.

**Tech Stack:** Same as Plan 1; no new top-level dependencies needed (vendor SDKs avoided in favor of plain `net/http`).

**Prerequisite:** Plan 1 merged. Working on the same worktree (or a successor branch off the foundation branch).

**Provider docs to consult while implementing:**

- AssemblyAI: https://www.assemblyai.com/docs/api-reference/transcripts
- ElevenLabs Speech-to-Text: https://elevenlabs.io/docs/api-reference/speech-to-text/convert
- OpenAI Whisper / `gpt-4o-transcribe`: https://platform.openai.com/docs/api-reference/audio/createTranscription
- Gemini Files API + multimodal generate: https://ai.google.dev/api/files (audio input)
- Mistral Voxtral: https://docs.mistral.ai/capabilities/audio/

For each provider, **use the `context7` MCP server before writing the client** to confirm the current request shape (endpoints, multipart fields, auth header). Provider APIs change; the references above are starting points.

---

## File map

```
internal/adapters/api/
├── assemblyai/   client.go  models.go  parse.go  parse_test.go  client_test.go  integration_test.go
├── elevenlabs/   ...
├── openai/       ...
├── gemini/       ...
└── mistral/      ...

testdata/
├── assemblyai_sample.json    # canned response per provider
├── elevenlabs_sample.json
├── openai_sample.json
├── gemini_sample.json
└── mistral_sample.json

internal/delivery/wire.go      # modified to register each provider
```

---

## Per-provider task template

Each of N1-N5 follows this same shape. The body of each task lists only the
provider-specific differences (endpoint, request encoding, response shape).

**Pattern (do this for every provider):**

1. Write `testdata/<provider>_sample.json` — minimal valid response captured
   from the real API or hand-written from the docs.
2. Write `internal/adapters/api/<provider>/models.go` with the static
   `modelCaps` map, `Models()`, `DefaultModel()`, `Capabilities()`. Include
   `AcceptedInputs` per the provider's documented format list.
3. TDD `parse.go` against the fixture: write `parse_test.go` first, run to
   confirm failure, implement `parse(data []byte, model string) (*domain.Result, error)`.
4. TDD `client.go` against `httptest.NewServer`: assert headers, multipart
   fields, parses the fixture back into a `*domain.Result`. Implement using
   `retry.Do` + `retry.HTTPError`.
5. Add `integration_test.go` with `//go:build integration`, gated on
   `TRANSCRIBE_<PROVIDER>_KEY`.
6. Commit each provider as one logical change set (or split parse vs client
   if it keeps commits small).

For details on the exact code shape, refer to `internal/adapters/api/groq/`
from Plan 1 — every provider mirrors it, only the wire format changes.

---

## Phase N — Add the five providers

### Task N1: AssemblyAI

**Provider specifics:**

- **Endpoint:** `https://api.assemblyai.com/v2/transcript` is the *creation* endpoint; AssemblyAI is **two-step**: POST audio to `/v2/upload`, then POST `{audio_url, ...}` to `/v2/transcript`, then poll `/v2/transcript/{id}` until status is `completed` or `error`.
- **Auth:** `Authorization: <api_key>` header (no `Bearer ` prefix).
- **MaxUploadBytes:** 200 MB.
- **Models:** `best` (default), `nano`.
- **AcceptedInputs:** mp3, wav, mp4, m4a, aac, flac, ogg, webm, mov, mpeg.
- **Word timestamps:** present in `words[]` with `start`/`end` in milliseconds.

**Files:**

- Create: `testdata/assemblyai_sample.json`
- Create: `internal/adapters/api/assemblyai/{models,parse,client}.go` + tests
- Create: `internal/adapters/api/assemblyai/integration_test.go`

- [ ] **Step 1: Capture / write the fixture**

Minimal `testdata/assemblyai_sample.json`:

```json
{
  "id": "abc",
  "status": "completed",
  "text": "Hello world",
  "language_code": "en",
  "audio_duration": 5,
  "words": [
    {"text": "Hello", "start": 100,  "end": 600},
    {"text": "world", "start": 700, "end": 1200}
  ]
}
```

- [ ] **Step 2: Write `models.go`**

Static map with `WordTimestamps: true`, `SegmentTimestamps: false`, `LanguageHint: true`, and `AcceptedInputs` per the format list above. `MaxUploadBytes = 200 * 1024 * 1024`.

- [ ] **Step 3: TDD `parse.go`**

Test:

```go
r, err := parse(fixture, "best")
require.Equal(t, "Hello world", r.Text)
require.Equal(t, "en", r.Language)
require.Equal(t, 5*time.Second, r.Duration)
require.Len(t, r.Words, 2)
require.Equal(t, 100*time.Millisecond, r.Words[0].Start)
require.Equal(t, 1200*time.Millisecond, r.Words[1].End)
```

Implement: unmarshal into the response struct shape above, convert `start`/`end` (ms) directly to `time.Duration` via `time.Duration(n)*time.Millisecond`.

- [ ] **Step 4: TDD `client.go`**

Test uses `httptest.NewServer` with **three routes**: `/v2/upload` returns `{"upload_url": <server>/audio/X}`, `/v2/transcript` returns `{"id":"abc"}` (status `queued`), `/v2/transcript/abc` returns the fixture. Client should: PUT bytes to upload, POST JSON `{audio_url, language_code, word_boost?, ...}`, poll the transcript ID with 1-second backoff until status `completed`. For the test, override the poll interval via a package-level `var pollInterval = 1 * time.Second` that the test sets to `1 * time.Millisecond`.

- [ ] **Step 5: Integration test**

```go
//go:build integration

func TestIntegration_AssemblyAI(t *testing.T) {
    key := os.Getenv("TRANSCRIBE_ASSEMBLYAI_KEY")
    if key == "" { t.Skip("...") }
    c := New(key, http.DefaultClient)
    res, err := c.Transcribe(context.Background(),
        domain.AudioFile{Path: "../../../../testdata/short-sample.mp3", Container: "mp3", Codec: "mp3"},
        ports.ProviderOpts{Model: c.DefaultModel(), Language: "en"})
    require.NoError(t, err)
    require.NotEmpty(t, res.Text)
}
```

- [ ] **Step 6: Commit**

```bash
git add internal/adapters/api/assemblyai/ testdata/assemblyai_sample.json
git commit -m "feat(assemblyai): provider adapter (upload + poll)"
```

---

### Task N2: ElevenLabs

**Provider specifics:**

- **Endpoint:** `POST https://api.elevenlabs.io/v1/speech-to-text` — single-shot multipart.
- **Auth:** `xi-api-key: <key>` header.
- **MaxUploadBytes:** 1 GB (`1000 * 1024 * 1024`).
- **Models:** `scribe_v1` (default), `scribe_v1_experimental`.
- **Multipart fields:** `file` (binary), `model_id`, optional `language_code`, `diarize=false` (v1 of this plan), `timestamps_granularity=word`.
- **AcceptedInputs:** mp3, mp4, m4a, wav, flac, ogg, webm, opus, mpga.
- **Word timestamps:** `words[]` with `start`/`end` in **seconds (float)** and `type` of `"word"`/`"spacing"` — filter out spacing entries.

**Files:**

- Create: `testdata/elevenlabs_sample.json`
- Create: `internal/adapters/api/elevenlabs/{models,parse,client}.go` + tests + integration

- [ ] **Step 1: Fixture**

```json
{
  "language_code": "eng",
  "language_probability": 0.99,
  "text": "Hello world",
  "words": [
    {"text": "Hello", "start": 0.1, "end": 0.6, "type": "word"},
    {"text": " ",     "start": 0.6, "end": 0.7, "type": "spacing"},
    {"text": "world", "start": 0.7, "end": 1.2, "type": "word"}
  ]
}
```

- [ ] **Step 2: `models.go`**

`scribe_v1` capability: `WordTimestamps: true`, `LanguageHint: true`. `AcceptedInputs` per docs.

- [ ] **Step 3: TDD `parse.go`** — drop entries where `type != "word"`. Convert float seconds via `time.Duration(s * float64(time.Second))`. Note: `language_code` is ISO-639-3 (`eng`); pass through verbatim.
- [ ] **Step 4: TDD `client.go`** — multipart with the fields listed above; single response.
- [ ] **Step 5: Integration test**
- [ ] **Step 6: Commit**

```bash
git commit -m "feat(elevenlabs): provider adapter"
```

---

### Task N3: OpenAI

**Provider specifics:**

- **Endpoint:** `POST https://api.openai.com/v1/audio/transcriptions`.
- **Auth:** `Authorization: Bearer <key>`.
- **MaxUploadBytes:** 25 MB.
- **Models:** `whisper-1` (default — has word timestamps via `timestamp_granularities[]=word`), `gpt-4o-transcribe` (also returns timestamps), `gpt-4o-mini-transcribe`.
- **Multipart fields:** `file`, `model`, `response_format=verbose_json`, `timestamp_granularities[]=word`, `timestamp_granularities[]=segment`, optional `language`.
- **AcceptedInputs:** mp3, mp4, mpeg, mpga, m4a, wav, webm, flac, ogg, opus.
- **Word timestamps:** `words[]` with `word`/`start`/`end` in **seconds (float)**.

**Files:** standard set.

- [ ] **Step 1: Fixture** (similar to Groq's — OpenAI's verbose_json is the schema Groq mirrors). Reuse the Groq fixture structure but bump the values to make the test independent.
- [ ] **Step 2: `models.go`** — three model entries. **Important:** if/when OpenAI ships a text-only audio model (e.g. some `gpt-4o-audio-preview` variants don't return timestamps), set `WordTimestamps: false` for that model so the capability check in the service rejects SRT outputs.
- [ ] **Step 3: TDD `parse.go`** — almost identical to Groq's; consider extracting a shared "OpenAI-compatible verbose JSON" helper if the duplication bites, but YAGNI for now.
- [ ] **Step 4: TDD `client.go`** — POST multipart, same retry policy.
- [ ] **Step 5: Integration test** with `TRANSCRIBE_OPENAI_KEY`.
- [ ] **Step 6: Commit**

```bash
git commit -m "feat(openai): provider adapter"
```

---

### Task N4: Gemini

**Provider specifics:**

- **Two-step like AssemblyAI:** upload via the Files API (`POST https://generativelanguage.googleapis.com/upload/v1beta/files`), then generate via `POST .../v1beta/models/<model>:generateContent` with `file_data: {mime_type, file_uri}` parts.
- **Auth:** `x-goog-api-key: <key>` header (newer pattern) or `?key=<key>` query string.
- **MaxUploadBytes:** 2 GB per file (Files API). Effectively unlimited for typical recordings.
- **Models:** `gemini-2.0-flash`, `gemini-2.5-flash`, `gemini-2.5-pro`. **Capability nuance:** Gemini does NOT natively return word-level timestamps — only the text. Set `WordTimestamps: false` for all current Gemini models. This means Gemini + SRT will be rejected by the service's capability check at submit time, exactly as the spec intends.
- **AcceptedInputs:** wav, mp3, aiff, aac, ogg, flac.
- **Prompt:** include something like "Transcribe the audio. Return only the spoken text, with no commentary." in the `contents` part.

**Files:** standard set.

- [ ] **Step 1: Fixture** — Gemini's response is a JSON envelope around free-form text; the fixture should be the smallest realistic shape (an array of `candidates` with `content.parts[].text`).

```json
{
  "candidates": [
    {"content": {"parts": [{"text": "Hello world"}]}, "finishReason": "STOP"}
  ]
}
```

- [ ] **Step 2: `models.go`** — three entries, all `WordTimestamps: false`. Document in a code comment that Gemini is text-only for the foreseeable future.
- [ ] **Step 3: TDD `parse.go`** — concatenate `text` across `candidates[0].content.parts`. No timestamps, no words array.
- [ ] **Step 4: TDD `client.go`** — test the upload step against `httptest.NewServer` and the generateContent step against the same server. Real Files API returns `{"file": {"uri": "..."}}`; client passes that uri into the generate call.
- [ ] **Step 5: Integration test** with `TRANSCRIBE_GEMINI_KEY`.
- [ ] **Step 6: Commit**

```bash
git commit -m "feat(gemini): provider adapter (files api + generate)"
```

---

### Task N5: Mistral (Voxtral)

**Provider specifics:**

- **Endpoint:** `POST https://api.mistral.ai/v1/audio/transcriptions` (Voxtral; see docs.mistral.ai/capabilities/audio/).
- **Auth:** `Authorization: Bearer <key>`.
- **MaxUploadBytes:** check docs — likely 25-50 MB. Use 25 MB to be safe.
- **Models:** `voxtral-mini-latest`, `voxtral-small-latest`.
- **Multipart fields:** `file`, `model`, optional `language`. **Word timestamps**: per docs, Voxtral returns segment-level timestamps but **not** word-level. Reflect in capabilities (`WordTimestamps: false, SegmentTimestamps: true`) — this means Voxtral + SRT/DaVinciSRT is rejected by the capability check. Consider whether to add a future `FormatSegmentSRT` (out of scope here).
- **AcceptedInputs:** mp3, wav, flac, ogg, opus, m4a, aac, webm.

**Files:** standard set.

- [ ] **Step 1: Fixture** — shape per Mistral docs; minimum is `{"text": "...", "language": "en", "segments": [...]}`.
- [ ] **Step 2: `models.go`** — both models with `WordTimestamps: false`, `SegmentTimestamps: true`. Comment that text-only is the v1 output.
- [ ] **Step 3: TDD `parse.go`** — populate `Result.Text`, `Result.Language`, `Result.Segments`. `Result.Words` stays empty.
- [ ] **Step 4: TDD `client.go`** — standard multipart POST.
- [ ] **Step 5: Integration test** with `TRANSCRIBE_MISTRAL_KEY`.
- [ ] **Step 6: Commit**

```bash
git commit -m "feat(mistral): voxtral provider adapter"
```

---

### Task N6: Register all providers in the composition root

**Files:**

- Modify: `internal/delivery/wire.go`

- [ ] **Step 1: Add the new imports and registrations**

```go
package delivery

import (
    "net/http"

    "github.com/leotulipan/transcribe/internal/adapters/api/assemblyai"
    "github.com/leotulipan/transcribe/internal/adapters/api/elevenlabs"
    "github.com/leotulipan/transcribe/internal/adapters/api/gemini"
    "github.com/leotulipan/transcribe/internal/adapters/api/groq"
    "github.com/leotulipan/transcribe/internal/adapters/api/mistral"
    "github.com/leotulipan/transcribe/internal/adapters/api/openai"
    "github.com/leotulipan/transcribe/internal/adapters/audio"
    "github.com/leotulipan/transcribe/internal/adapters/cache"
    "github.com/leotulipan/transcribe/internal/adapters/format"
    "github.com/leotulipan/transcribe/internal/core/domain"
    "github.com/leotulipan/transcribe/internal/core/services"
    "github.com/leotulipan/transcribe/internal/ports"
)

func BuildService(cfg ports.Config, log ports.Logger) (ports.TranscribeService, error) {
    ffmpeg, err := audio.New(cfg.FFmpegPath, "", log)
    if err != nil {
        return nil, err
    }
    httpClient := &http.Client{}

    providers := map[domain.ProviderID]ports.Provider{}
    if k := cfg.APIKeys[domain.ProviderGroq];       k != "" { providers[domain.ProviderGroq]       = groq.New(k, httpClient) }
    if k := cfg.APIKeys[domain.ProviderOpenAI];     k != "" { providers[domain.ProviderOpenAI]     = openai.New(k, httpClient) }
    if k := cfg.APIKeys[domain.ProviderAssemblyAI]; k != "" { providers[domain.ProviderAssemblyAI] = assemblyai.New(k, httpClient) }
    if k := cfg.APIKeys[domain.ProviderElevenLabs]; k != "" { providers[domain.ProviderElevenLabs] = elevenlabs.New(k, httpClient) }
    if k := cfg.APIKeys[domain.ProviderGemini];     k != "" { providers[domain.ProviderGemini]     = gemini.New(k, httpClient) }
    if k := cfg.APIKeys[domain.ProviderMistral];    k != "" { providers[domain.ProviderMistral]    = mistral.New(k, httpClient) }

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
go build ./...
```

- [ ] **Step 3: Run all tests with -race**

```bash
go test -race ./...
```

Expected: PASS.

- [ ] **Step 4: Smoke-test `providers` subcommand**

```powershell
$env:TRANSCRIBE_GROQ_KEY = "test"
$env:TRANSCRIBE_OPENAI_KEY = "test"
./bin/transcribe.exe providers
```

Expected: groq and openai listed (with the rest empty until their keys are also set).

- [ ] **Step 5: Commit**

```bash
git add internal/delivery/wire.go
git commit -m "feat(delivery): register all six providers"
```

---

## Self-review

(note all api keys from C:\Users\leona\.transcribe\.env can be used)

- [ ] `go vet ./...` — clean
- [ ] `go test -race ./...` — all pass
- [ ] All five new provider unit-test suites have a happy-path test against their fixture and at least one error-path test (non-2xx, malformed JSON).
- [ ] All five `integration_test.go` files use the build tag `integration` and `t.Skip` when their key env var is unset.
- [ ] With a real OpenAI key: end-to-end CLI call against `testdata/short-sample.mp3` with `--api openai --output text,srt` produces both files.
- [ ] With a real Gemini key: `--api gemini --output srt` produces an `ErrIncompatible` (Gemini has no word timestamps).
- [ ] `providers --json` lists all six configured providers when all keys are set.

Plan 2 is complete when the matrix of (provider × text/srt/davinci.srt) compiles and the timestamps-required combinations fail safely with `ErrIncompatible`.
