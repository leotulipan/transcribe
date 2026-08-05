# AssemblyAI Auto Model Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Default AssemblyAI jobs to omitting `speech_models` so the API always picks its current flagship model, before `universal-3-pro` starts erroring on 2026-09-02.

**Architecture:** An `auto` sentinel model becomes the AssemblyAI default. When it is selected and the user supplied no explicit fallback list, `buildSpeechModels` returns `nil` and `submitTranscript` omits the `speech_models` key entirely. Explicit pins keep today's behavior (chosen models plus a `universal-2` safety net). A pure `NormalizeModels` function rewrites the retired `universal-3-pro` to `universal-3-5-pro`, and the client logs a one-line deprecation warning through an injected `ports.Logger`.

**Tech Stack:** Go 1.x, standard library `net/http` and `encoding/json`, `github.com/stretchr/testify/require` for assertions, `httptest` for server fakes.

**Spec:** `docs/superpowers/specs/2026-08-02-assemblyai-universal-3-5-design.md`

## Global Constraints

- All work is confined to `internal/adapters/api/assemblyai/`, one line of `internal/delivery/wire.go`, one string in `internal/delivery/gui/mainwindow.go`, and documentation. No other provider changes behavior.
- The only model IDs AssemblyAI's `speech_models` array accepts are `universal-3-5-pro` and `universal-2`. Never advertise or send anything else, except that the `auto` sentinel is never sent at all.
- `auto` is a local sentinel, not an API value. It must never appear inside a `speech_models` array in a request body.
- Deprecated ID mapping is exactly `universal-3-pro` → `universal-3-5-pro`. Unknown IDs pass through untouched so real typos still surface as API errors.
- The deprecation date to cite in user-facing copy is **2026-09-02**.
- `NormalizeModels` must be pure and idempotent — calling it twice yields the same result and reports no remap the second time.
- Explicit pins still get `universal-2` appended as a safety net. Only the auto path omits the field.
- `internal/delivery/tui/wizard.go:43` calls `assemblyai.New(key, nil)`. Its signature must keep working unchanged.

### Running tests

Per-package tests run with plain `go test` because this package is pure Go:

```bash
go test ./internal/adapters/api/assemblyai/ -run TestName -v
```

A full `go test ./...` also builds the Fyne GUI package, which needs cgo and `gcc` on PATH. If `go test ./...` fails with a cgo/gcc error, that is an environment limitation and not a regression from this work — use `scripts/build.ps1`, which sets up the cgo toolchain. Per-package tests are the primary TDD loop for every task here.

---

### Task 1: Model catalog and deprecation normalization

Introduces the `auto` sentinel, swaps `universal-3-pro` for `universal-3-5-pro`, and adds the pure `NormalizeModels` helper the later tasks build on.

**Files:**
- Modify: `internal/adapters/api/assemblyai/models.go` (whole file rewritten)
- Test: `internal/adapters/api/assemblyai/models_test.go` (whole file rewritten)

**Interfaces:**
- Consumes: nothing (first task).
- Produces:
  - `const AutoModel = "auto"` — exported sentinel.
  - `const fallbackModel = "universal-2"` — unexported, unchanged name.
  - `func Models() []string` → `["auto", "universal-3-5-pro", "universal-2"]`
  - `func DefaultModel() string` → `"auto"`
  - `func Capabilities(model string) ports.ModelCapabilities` — unchanged signature.
  - `func NormalizeModels(in []string) (out, remapped []string)` — `out` is `in` with deprecated IDs replaced, order preserved, same length; `remapped` holds the **original** deprecated IDs that were replaced, in encounter order, and is `nil` when nothing was remapped. A `nil` or empty `in` yields `nil, nil`.

- [ ] **Step 1: Write the failing tests**

Replace the entire contents of `internal/adapters/api/assemblyai/models_test.go` with:

```go
package assemblyai

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

func TestDefaultModel(t *testing.T) {
	// The default is the "auto" sentinel: it makes the client omit
	// speech_models so AssemblyAI picks its current flagship model.
	require.Equal(t, AutoModel, DefaultModel())
	require.Equal(t, "auto", AutoModel)
}

func TestModels_PriorityOrder(t *testing.T) {
	// First entry must be the default so UIs that pick models[0] still get
	// the recommended selection.
	got := Models()
	require.NotEmpty(t, got)
	require.Equal(t, DefaultModel(), got[0])
	// Besides the sentinel, only the values AssemblyAI's speech_models array
	// accepts may be advertised; everything else 400s. See models.go.
	require.Equal(t, []string{"auto", "universal-3-5-pro", "universal-2"}, got)
	// Retired on 2026-09-02 — must not be offered any more.
	require.NotContains(t, got, "universal-3-pro")
	require.NotContains(t, got, "universal-3")
	require.NotContains(t, got, "slam-1")
	require.NotContains(t, got, "best")
	require.NotContains(t, got, "nano")
}

func TestCapabilities_AutoHasWordTimestamps(t *testing.T) {
	// The GUI greys out the SRT checkboxes when the selected model reports no
	// word timestamps. "auto" is the default selection, so it must report the
	// full capability set or the default UI state would be wrong.
	c := Capabilities(AutoModel)
	require.True(t, c.WordTimestamps)
	require.False(t, c.SegmentTimestamps)
	require.True(t, c.Diarization)
	require.True(t, c.LanguageHint)
	require.Contains(t, c.AcceptedInputs, domain.AudioFormat{Codec: "mp3"})
	require.Contains(t, c.AcceptedInputs, domain.AudioFormat{Codec: "flac"})
}

func TestCapabilities_Universal35ProHasWordTimestamps(t *testing.T) {
	c := Capabilities("universal-3-5-pro")
	require.True(t, c.WordTimestamps)
	require.True(t, c.Diarization)
	require.True(t, c.LanguageHint)
	require.Contains(t, c.AcceptedInputs, domain.AudioFormat{Codec: "mp3"})
}

func TestCapabilities_Universal2Works(t *testing.T) {
	c := Capabilities("universal-2")
	require.True(t, c.WordTimestamps)
	require.True(t, c.LanguageHint)
}

func TestCapabilities_RetiredModelReturnsZero(t *testing.T) {
	// universal-3-pro is retired, and "best"/"nano" were never accepted by the
	// plural array. All are treated as unknown here; the request path remaps
	// universal-3-pro before it ever reaches Capabilities.
	for _, m := range []string{"universal-3-pro", "best", "nano"} {
		c := Capabilities(m)
		require.False(t, c.WordTimestamps, "model %q must be unknown", m)
		require.Empty(t, c.AcceptedInputs, "model %q must be unknown", m)
	}
}

func TestCapabilities_UnknownModelReturnsZero(t *testing.T) {
	c := Capabilities("totally-made-up")
	require.False(t, c.WordTimestamps)
	require.Empty(t, c.AcceptedInputs)
}

func TestNormalizeModels_RemapsDeprecated(t *testing.T) {
	out, remapped := NormalizeModels([]string{"universal-3-pro"})
	require.Equal(t, []string{"universal-3-5-pro"}, out)
	require.Equal(t, []string{"universal-3-pro"}, remapped)
}

func TestNormalizeModels_PreservesOrderAndUnknowns(t *testing.T) {
	out, remapped := NormalizeModels([]string{"universal-2", "universal-3-pro", "made-up"})
	require.Equal(t, []string{"universal-2", "universal-3-5-pro", "made-up"}, out)
	require.Equal(t, []string{"universal-3-pro"}, remapped)
}

func TestNormalizeModels_NothingToDo(t *testing.T) {
	out, remapped := NormalizeModels([]string{"universal-3-5-pro", "universal-2"})
	require.Equal(t, []string{"universal-3-5-pro", "universal-2"}, out)
	require.Nil(t, remapped)
}

func TestNormalizeModels_Idempotent(t *testing.T) {
	once, _ := NormalizeModels([]string{"universal-3-pro", "universal-2"})
	twice, remapped := NormalizeModels(once)
	require.Equal(t, once, twice)
	require.Nil(t, remapped, "second pass must report no remapping")
}

func TestNormalizeModels_Empty(t *testing.T) {
	out, remapped := NormalizeModels(nil)
	require.Nil(t, out)
	require.Nil(t, remapped)

	out, remapped = NormalizeModels([]string{})
	require.Empty(t, out)
	require.Nil(t, remapped)
}

func TestNormalizeModels_DoesNotMutateInput(t *testing.T) {
	in := []string{"universal-3-pro"}
	_, _ = NormalizeModels(in)
	require.Equal(t, []string{"universal-3-pro"}, in, "input slice must not be mutated")
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./internal/adapters/api/assemblyai/ -run 'TestDefaultModel|TestModels_PriorityOrder|TestCapabilities|TestNormalizeModels' -v`

Expected: FAIL — compile error `undefined: AutoModel` and `undefined: NormalizeModels`.

- [ ] **Step 3: Write the implementation**

Replace the entire contents of `internal/adapters/api/assemblyai/models.go` with:

```go
package assemblyai

import (
	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

// Per https://www.assemblyai.com/docs/api-reference/transcripts
// AssemblyAI uses a two-step flow: upload then poll.
// Word timestamps are milliseconds in the response.
//
// Model selection: AssemblyAI's `speech_models` array accepts only
// `universal-3-5-pro` and `universal-2`. Anything else — including the
// retired `universal-3-pro`, the non-pro `universal-3`, `slam-1`, and the
// legacy singular `speech_model` values `best`/`nano` — is rejected with
// HTTP 400.
//
// Omitting `speech_models` entirely is equivalent to sending
// ["universal-3-5-pro", "universal-2"]: AssemblyAI runs its current flagship
// and falls back to universal-2 for languages outside the flagship's 18. That
// is what AutoModel expresses, and it is the default so we never have to ship
// a code change when AssemblyAI promotes a new flagship.
// See https://www.assemblyai.com/docs/pre-recorded-audio/select-the-speech-model

// AutoModel is a local sentinel, never an API value. When it is the selected
// model and no explicit fallback list was given, the client omits the
// speech_models key from the request body.
const AutoModel = "auto"

// deprecatedModels maps retired model IDs to their replacement. AssemblyAI
// retires universal-3-pro on 2026-09-02; requests naming it error after that.
var deprecatedModels = map[string]string{
	"universal-3-pro": "universal-3-5-pro",
}

var sharedInputs = []domain.AudioFormat{
	{Codec: "mp3"}, {Codec: "wav"}, {Codec: "mp4"}, {Codec: "m4a"},
	{Codec: "aac"}, {Codec: "flac"}, {Codec: "ogg"}, {Codec: "webm"},
	{Codec: "mov"}, {Codec: "mpeg"},
}

var sttCaps = ports.ModelCapabilities{
	WordTimestamps:    true,
	SegmentTimestamps: false,
	Diarization:       true,
	LanguageHint:      true,
	AcceptedInputs:    sharedInputs,
}

// AutoModel advertises the same capabilities as the concrete models because
// the whole Universal family is uniform on these axes; if that ever stops
// being true, auto must advertise the intersection.
var modelCaps = map[string]ports.ModelCapabilities{
	AutoModel:           sttCaps,
	"universal-3-5-pro": sttCaps,
	"universal-2":       sttCaps,
}

// Models returns the supported model IDs with the default first, so UIs that
// pick models[0] land on the recommended selection. `auto` lets AssemblyAI
// choose; `universal-3-5-pro` covers 18 languages at the highest accuracy;
// `universal-2` covers 99 languages and is the automatic fallback. Map
// iteration would be random, hence the explicit slice.
func Models() []string {
	return []string{
		AutoModel,
		"universal-3-5-pro",
		"universal-2",
	}
}

func DefaultModel() string { return AutoModel }

// fallbackModel is appended to every explicitly pinned speech_models array so
// a job still succeeds when the pinned model does not cover the audio's
// language. The GUI/TUI have no fallback picker, so this guarantees a
// resilient result for every explicit selection.
const fallbackModel = "universal-2"

// NormalizeModels rewrites deprecated model IDs, preserving order and length.
// It returns the cleaned list plus the original deprecated IDs that were
// replaced (nil when there were none), so callers can warn the user. Unknown
// IDs pass through untouched: AssemblyAI's 400 response already names the
// valid set, and silently rewriting an unrecognized string would hide typos.
// Pure and idempotent — safe to call from several layers.
func NormalizeModels(in []string) (out, remapped []string) {
	if len(in) == 0 {
		return in, nil
	}
	out = make([]string, len(in))
	for i, m := range in {
		if replacement, ok := deprecatedModels[m]; ok {
			out[i] = replacement
			remapped = append(remapped, m)
			continue
		}
		out[i] = m
	}
	return out, remapped
}

func Capabilities(model string) ports.ModelCapabilities {
	return modelCaps[model] // zero value if unknown — fail-safe
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test ./internal/adapters/api/assemblyai/ -run 'TestDefaultModel|TestModels_PriorityOrder|TestCapabilities|TestNormalizeModels' -v`

Expected: PASS for all of them.

Note: other tests in this package still reference the old default and will fail until Task 2. Do not run the whole package yet.

- [ ] **Step 5: Commit**

```bash
git add internal/adapters/api/assemblyai/models.go internal/adapters/api/assemblyai/models_test.go
git commit -m "feat(assemblyai): add auto model sentinel and deprecation remap"
```

---

### Task 2: Omit speech_models in auto mode

Makes the request path honor the sentinel: auto with no explicit pins sends no `speech_models` key at all, while explicit pins behave exactly as before (after normalization).

**Files:**
- Modify: `internal/adapters/api/assemblyai/client.go:176-197` (`buildSpeechModels`) and `client.go:199-222` (`submitTranscript` body assembly)
- Test: `internal/adapters/api/assemblyai/client_test.go:197-261` (replace three existing tests, add two)

**Interfaces:**
- Consumes: `AutoModel`, `fallbackModel`, `NormalizeModels` from Task 1.
- Produces: `func buildSpeechModels(primary string, fallbacks []string) []string` — returns `nil` when the request should omit the key; otherwise the ordered, deduped, normalized list. Task 3 wraps this call site with logging.

- [ ] **Step 1: Write the failing tests**

In `internal/adapters/api/assemblyai/client_test.go`, replace the three tests spanning lines 197-261 (`TestAssemblyAI_RequestIncludesSpeechModels`, `TestAssemblyAI_SpeechModelsDefaultsToFallback`, `TestAssemblyAI_SpeechModelsNoDuplicateFallback`) with:

```go
func TestAssemblyAI_RequestIncludesSpeechModels(t *testing.T) {
	// An explicit fallback list is sent verbatim, with universal-2 as the
	// safety net (already last here, so no duplicate is added).
	var got map[string]interface{}
	srv := captureTranscriptPayload(t, &got)
	defer srv.Close()

	audioPath := filepath.Join(t.TempDir(), "tiny.mp3")
	require.NoError(t, os.WriteFile(audioPath, []byte("\xff\xfb\x90\x00"), 0o644))

	c := NewWithEndpoint("test-key", srv.URL, http.DefaultClient)
	_, err := c.Transcribe(context.Background(),
		domain.AudioFile{Path: audioPath, Container: "mp3", Codec: "mp3", SizeBytes: 4},
		ports.ProviderOpts{Model: AutoModel, SpeechModels: []string{"universal-3-5-pro", "universal-2"}},
	)
	require.NoError(t, err)
	raw, ok := got["speech_models"].([]interface{})
	require.True(t, ok, "speech_models must be an array")
	require.Equal(t, []interface{}{"universal-3-5-pro", "universal-2"}, raw)
}

func TestAssemblyAI_AutoModelOmitsSpeechModels(t *testing.T) {
	// The whole point of the auto sentinel: no speech_models key at all, so
	// AssemblyAI applies its own current default and fallback chain.
	var got map[string]interface{}
	srv := captureTranscriptPayload(t, &got)
	defer srv.Close()

	audioPath := filepath.Join(t.TempDir(), "tiny.mp3")
	require.NoError(t, os.WriteFile(audioPath, []byte("\xff\xfb\x90\x00"), 0o644))

	c := NewWithEndpoint("test-key", srv.URL, http.DefaultClient)
	_, err := c.Transcribe(context.Background(),
		domain.AudioFile{Path: audioPath, Container: "mp3", Codec: "mp3", SizeBytes: 4},
		ports.ProviderOpts{Model: AutoModel},
	)
	require.NoError(t, err)

	_, hasPlural := got["speech_models"]
	require.False(t, hasPlural, "auto mode must omit speech_models entirely")
	_, hasSingular := got["speech_model"]
	require.False(t, hasSingular, "deprecated singular speech_model must not be sent")
	// The rest of the body is unaffected.
	require.NotEmpty(t, got["audio_url"])
}

func TestAssemblyAI_EmptyModelOmitsSpeechModels(t *testing.T) {
	// Defensive: the pipeline substitutes DefaultModel() for an empty model,
	// but a direct caller may still pass "". That must behave like auto, not
	// send an empty-string model.
	var got map[string]interface{}
	srv := captureTranscriptPayload(t, &got)
	defer srv.Close()

	audioPath := filepath.Join(t.TempDir(), "tiny.mp3")
	require.NoError(t, os.WriteFile(audioPath, []byte("\xff\xfb\x90\x00"), 0o644))

	c := NewWithEndpoint("test-key", srv.URL, http.DefaultClient)
	_, err := c.Transcribe(context.Background(),
		domain.AudioFile{Path: audioPath, Container: "mp3", Codec: "mp3", SizeBytes: 4},
		ports.ProviderOpts{Model: ""},
	)
	require.NoError(t, err)
	_, hasPlural := got["speech_models"]
	require.False(t, hasPlural, "empty model must behave like auto")
}

func TestAssemblyAI_PinnedModelGetsFallbackAppended(t *testing.T) {
	// An explicitly pinned model keeps the universal-2 safety net.
	var got map[string]interface{}
	srv := captureTranscriptPayload(t, &got)
	defer srv.Close()

	audioPath := filepath.Join(t.TempDir(), "tiny.mp3")
	require.NoError(t, os.WriteFile(audioPath, []byte("\xff\xfb\x90\x00"), 0o644))

	c := NewWithEndpoint("test-key", srv.URL, http.DefaultClient)
	_, err := c.Transcribe(context.Background(),
		domain.AudioFile{Path: audioPath, Container: "mp3", Codec: "mp3", SizeBytes: 4},
		ports.ProviderOpts{Model: "universal-3-5-pro"},
	)
	require.NoError(t, err)

	_, hasSingular := got["speech_model"]
	require.False(t, hasSingular, "deprecated singular speech_model must not be sent")

	raw, ok := got["speech_models"].([]interface{})
	require.True(t, ok, "speech_models must be an array")
	require.Equal(t, []interface{}{"universal-3-5-pro", "universal-2"}, raw)
}

func TestAssemblyAI_DeprecatedModelIsRemapped(t *testing.T) {
	// A stale config saying universal-3-pro must still work after 2026-09-02.
	var got map[string]interface{}
	srv := captureTranscriptPayload(t, &got)
	defer srv.Close()

	audioPath := filepath.Join(t.TempDir(), "tiny.mp3")
	require.NoError(t, os.WriteFile(audioPath, []byte("\xff\xfb\x90\x00"), 0o644))

	c := NewWithEndpoint("test-key", srv.URL, http.DefaultClient)
	_, err := c.Transcribe(context.Background(),
		domain.AudioFile{Path: audioPath, Container: "mp3", Codec: "mp3", SizeBytes: 4},
		ports.ProviderOpts{Model: "universal-3-pro"},
	)
	require.NoError(t, err)
	raw, ok := got["speech_models"].([]interface{})
	require.True(t, ok)
	require.Equal(t, []interface{}{"universal-3-5-pro", "universal-2"}, raw)
	require.NotContains(t, raw, "universal-3-pro", "retired ID must never be sent")
}

func TestAssemblyAI_SpeechModelsNoDuplicateFallback(t *testing.T) {
	// When the selected model already is the fallback, it must not be duplicated.
	var got map[string]interface{}
	srv := captureTranscriptPayload(t, &got)
	defer srv.Close()

	audioPath := filepath.Join(t.TempDir(), "tiny.mp3")
	require.NoError(t, os.WriteFile(audioPath, []byte("\xff\xfb\x90\x00"), 0o644))

	c := NewWithEndpoint("test-key", srv.URL, http.DefaultClient)
	_, err := c.Transcribe(context.Background(),
		domain.AudioFile{Path: audioPath, Container: "mp3", Codec: "mp3", SizeBytes: 4},
		ports.ProviderOpts{Model: "universal-2"},
	)
	require.NoError(t, err)
	raw, ok := got["speech_models"].([]interface{})
	require.True(t, ok)
	require.Equal(t, []interface{}{"universal-2"}, raw)
}

func TestBuildSpeechModels(t *testing.T) {
	tests := []struct {
		name      string
		primary   string
		fallbacks []string
		want      []string
	}{
		{"auto alone omits", AutoModel, nil, nil},
		{"empty alone omits", "", nil, nil},
		{"auto with explicit pins uses the pins", AutoModel, []string{"universal-3-5-pro"}, []string{"universal-3-5-pro", "universal-2"}},
		{"deprecated primary is remapped", "universal-3-pro", nil, []string{"universal-3-5-pro", "universal-2"}},
		{"deprecated in fallbacks is remapped", "", []string{"universal-3-pro", "universal-2"}, []string{"universal-3-5-pro", "universal-2"}},
		{"fallback model alone is not duplicated", "universal-2", nil, []string{"universal-2"}},
		{"pinned model gets the safety net", "universal-3-5-pro", nil, []string{"universal-3-5-pro", "universal-2"}},
		{"empty entries are dropped", "", []string{"", "universal-3-5-pro"}, []string{"universal-3-5-pro", "universal-2"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := buildSpeechModels(tt.primary, tt.fallbacks)
			require.Equal(t, tt.want, got)
		})
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./internal/adapters/api/assemblyai/ -run 'TestAssemblyAI_AutoModelOmitsSpeechModels|TestBuildSpeechModels' -v`

Expected: FAIL — `TestAssemblyAI_AutoModelOmitsSpeechModels` fails because `speech_models` is present with `["auto","universal-2"]`, and `TestBuildSpeechModels` fails its `auto alone omits` case for the same reason.

- [ ] **Step 3: Write the implementation**

In `internal/adapters/api/assemblyai/client.go`, replace `buildSpeechModels` (currently lines 176-197) with:

```go
// buildSpeechModels assembles AssemblyAI's ordered `speech_models` array, or
// returns nil to signal that the key must be omitted entirely.
//
// Omission is the default path: with the AutoModel sentinel and no explicit
// fallback list, AssemblyAI applies its own current default chain
// (universal-3-5-pro then universal-2) so we never ship a code change when
// they promote a new flagship.
//
// When the caller does pin models, we keep the historical behavior: start
// from the explicit fallback list (else the single selected model), always
// append fallbackModel last as a language-coverage safety net, and dedupe
// while preserving order. Deprecated IDs are remapped first so a stale config
// never reaches the API. AssemblyAI's official param is the plural array; the
// singular `speech_model` is no longer sent.
func buildSpeechModels(primary string, fallbacks []string) []string {
	normalizedPrimary, _ := NormalizeModels([]string{primary})
	primary = normalizedPrimary[0]
	fallbacks, _ = NormalizeModels(fallbacks)

	if len(fallbacks) == 0 && (primary == AutoModel || primary == "") {
		return nil // omit speech_models — let AssemblyAI choose
	}

	src := fallbacks
	if len(src) == 0 {
		src = []string{primary}
	}
	src = append(append([]string{}, src...), fallbackModel)
	seen := make(map[string]bool, len(src))
	out := make([]string, 0, len(src))
	for _, m := range src {
		if m == "" || m == AutoModel || seen[m] {
			continue
		}
		seen[m] = true
		out = append(out, m)
	}
	return out
}
```

Then in `submitTranscript`, replace the body-literal assignment (currently lines 201-205) with:

```go
	body := map[string]interface{}{
		"audio_url":     audioURL,
		"language_code": opts.Language,
	}
	if models := buildSpeechModels(model, opts.SpeechModels); len(models) > 0 {
		body["speech_models"] = models
	}
```

The remaining lines of `submitTranscript` (the `language_code` deletion, `speaker_labels`, `keyterms_prompt`, marshalling, and the retry loop) stay exactly as they are.

- [ ] **Step 4: Run the full package tests**

Run: `go test ./internal/adapters/api/assemblyai/ -v`

Expected: PASS — every test in the package, including Task 1's, now that the old `universal-3-pro` expectations are gone.

- [ ] **Step 5: Commit**

```bash
git add internal/adapters/api/assemblyai/client.go internal/adapters/api/assemblyai/client_test.go
git commit -m "feat(assemblyai): omit speech_models in auto mode"
```

---

### Task 3: Deprecation warning through an injected logger

Tells the user their config is stale. Placing this in the adapter means one implementation covers the CLI, GUI, TUI, and batch templates.

**Files:**
- Modify: `internal/adapters/api/assemblyai/client.go:32-49` (struct and constructors) and the `submitTranscript` body assembly from Task 2
- Modify: `internal/delivery/wire.go:31`
- Test: `internal/adapters/api/assemblyai/client_test.go` (append)

**Interfaces:**
- Consumes: `NormalizeModels` (Task 1), `buildSpeechModels` (Task 2).
- Produces: `func (c *Client) WithLogger(l ports.Logger) *Client` — nil-safe chained setter returning the receiver, so `assemblyai.New(k, h).WithLogger(log)` is a single expression. A `Client` with no logger works exactly as before.

- [ ] **Step 1: Write the failing test**

Append to `internal/adapters/api/assemblyai/client_test.go`:

```go
// recordingLogger captures Warn calls so tests can assert on user-facing
// deprecation notices.
type recordingLogger struct{ warns []string }

func (l *recordingLogger) Debug(msg string, kv ...any) {}
func (l *recordingLogger) Info(msg string, kv ...any)  {}
func (l *recordingLogger) Warn(msg string, kv ...any)  { l.warns = append(l.warns, msg) }
func (l *recordingLogger) Error(msg string, kv ...any) {}

func TestAssemblyAI_WarnsOnDeprecatedModel(t *testing.T) {
	var got map[string]interface{}
	srv := captureTranscriptPayload(t, &got)
	defer srv.Close()

	audioPath := filepath.Join(t.TempDir(), "tiny.mp3")
	require.NoError(t, os.WriteFile(audioPath, []byte("\xff\xfb\x90\x00"), 0o644))

	log := &recordingLogger{}
	c := NewWithEndpoint("test-key", srv.URL, http.DefaultClient).WithLogger(log)
	_, err := c.Transcribe(context.Background(),
		domain.AudioFile{Path: audioPath, Container: "mp3", Codec: "mp3", SizeBytes: 4},
		ports.ProviderOpts{Model: "universal-3-pro"},
	)
	require.NoError(t, err)

	require.Len(t, log.warns, 1, "exactly one deprecation warning per request")
	require.Contains(t, log.warns[0], "universal-3-pro")
	require.Contains(t, log.warns[0], "universal-3-5-pro")
	require.Contains(t, log.warns[0], "2026-09-02")
}

func TestAssemblyAI_NoWarnForCurrentModel(t *testing.T) {
	var got map[string]interface{}
	srv := captureTranscriptPayload(t, &got)
	defer srv.Close()

	audioPath := filepath.Join(t.TempDir(), "tiny.mp3")
	require.NoError(t, os.WriteFile(audioPath, []byte("\xff\xfb\x90\x00"), 0o644))

	log := &recordingLogger{}
	c := NewWithEndpoint("test-key", srv.URL, http.DefaultClient).WithLogger(log)
	_, err := c.Transcribe(context.Background(),
		domain.AudioFile{Path: audioPath, Container: "mp3", Codec: "mp3", SizeBytes: 4},
		ports.ProviderOpts{Model: AutoModel},
	)
	require.NoError(t, err)
	require.Empty(t, log.warns)
}

func TestAssemblyAI_NilLoggerIsSafe(t *testing.T) {
	// tui/wizard.go constructs a client without a logger; a deprecated model
	// must remap silently rather than panic.
	var got map[string]interface{}
	srv := captureTranscriptPayload(t, &got)
	defer srv.Close()

	audioPath := filepath.Join(t.TempDir(), "tiny.mp3")
	require.NoError(t, os.WriteFile(audioPath, []byte("\xff\xfb\x90\x00"), 0o644))

	c := NewWithEndpoint("test-key", srv.URL, http.DefaultClient)
	_, err := c.Transcribe(context.Background(),
		domain.AudioFile{Path: audioPath, Container: "mp3", Codec: "mp3", SizeBytes: 4},
		ports.ProviderOpts{Model: "universal-3-pro"},
	)
	require.NoError(t, err)
	raw, ok := got["speech_models"].([]interface{})
	require.True(t, ok)
	require.Equal(t, []interface{}{"universal-3-5-pro", "universal-2"}, raw)
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./internal/adapters/api/assemblyai/ -run 'TestAssemblyAI_WarnsOnDeprecatedModel|TestAssemblyAI_NoWarnForCurrentModel|TestAssemblyAI_NilLoggerIsSafe' -v`

Expected: FAIL — compile error `c.WithLogger undefined (type *Client has no field or method WithLogger)`.

- [ ] **Step 3: Write the implementation**

In `internal/adapters/api/assemblyai/client.go`, add the `log` field to the struct (currently lines 34-38):

```go
type Client struct {
	apiKey   string
	endpoint string
	http     *http.Client
	log      ports.Logger // optional; nil disables warnings
}
```

Add the setter immediately after `NewWithEndpoint`:

```go
// WithLogger attaches a logger for non-fatal notices such as model
// deprecation warnings. Optional: a nil logger simply silences them, which is
// why callers like the TUI wizard can skip it.
func (c *Client) WithLogger(l ports.Logger) *Client {
	c.log = l
	return c
}
```

Then in `submitTranscript`, replace the Task 2 body assembly with a version that warns:

```go
	body := map[string]interface{}{
		"audio_url":     audioURL,
		"language_code": opts.Language,
	}
	c.warnDeprecated(model, opts.SpeechModels)
	if models := buildSpeechModels(model, opts.SpeechModels); len(models) > 0 {
		body["speech_models"] = models
	}
```

And add the helper next to `buildSpeechModels`:

```go
// warnDeprecated logs one notice per request when the caller named a retired
// model. The request itself still succeeds — buildSpeechModels remaps the ID —
// but the user needs to know their config is stale.
func (c *Client) warnDeprecated(primary string, fallbacks []string) {
	if c.log == nil {
		return
	}
	_, remapped := NormalizeModels(append([]string{primary}, fallbacks...))
	if len(remapped) == 0 {
		return
	}
	seen := make(map[string]bool, len(remapped))
	for _, old := range remapped {
		if seen[old] {
			continue
		}
		seen[old] = true
		c.log.Warn(fmt.Sprintf(
			"%s is deprecated (AssemblyAI retires it 2026-09-02); using %s. Update your config or --speech-models.",
			old, deprecatedModels[old],
		))
	}
}
```

`fmt` is already imported in `client.go`; `ports` is too.

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test ./internal/adapters/api/assemblyai/ -v`

Expected: PASS for the whole package.

- [ ] **Step 5: Wire the logger in**

In `internal/delivery/wire.go`, change line 31 from:

```go
	if k := cfg.APIKeys[domain.ProviderAssemblyAI]; k != "" { providers[domain.ProviderAssemblyAI] = assemblyai.New(k, httpClient) }
```

to:

```go
	if k := cfg.APIKeys[domain.ProviderAssemblyAI]; k != "" { providers[domain.ProviderAssemblyAI] = assemblyai.New(k, httpClient).WithLogger(log) }
```

Leave `internal/delivery/tui/wizard.go:43` untouched — it intentionally builds a logger-less client for key validation.

- [ ] **Step 6: Verify the wiring compiles**

Run: `go build ./internal/delivery/ ./internal/adapters/...`

Expected: no output (success). If this errors with a cgo/gcc message it is picking up the GUI package — build `./internal/delivery` alone, which is pure Go.

- [ ] **Step 7: Commit**

```bash
git add internal/adapters/api/assemblyai/client.go internal/adapters/api/assemblyai/client_test.go internal/delivery/wire.go
git commit -m "feat(assemblyai): warn when a deprecated model is remapped"
```

---

### Task 4: Record the model AssemblyAI actually used

Without this, every auto-mode sidecar JSON just says `"auto"` and there is no way to tell after the fact which model produced a transcript.

**Files:**
- Modify: `internal/adapters/api/assemblyai/parse.go:13-21` (response struct) and `parse.go:30-45` (the `parse` function)
- Test: `internal/adapters/api/assemblyai/parse_test.go` (append)

**Interfaces:**
- Consumes: `AutoModel` (Task 1).
- Produces: `func parse(data []byte, model string) (*domain.Result, error)` — signature unchanged. `Result.Model` now holds the API's echoed `speech_model` when `model == AutoModel` and the echo is non-empty; otherwise the requested model, as before.

- [ ] **Step 1: Write the failing tests**

Append to `internal/adapters/api/assemblyai/parse_test.go`:

```go
func TestParse_AutoRecordsActualModel(t *testing.T) {
	// In auto mode we never told the API which model to use, so the result
	// must name whatever it actually ran rather than the "auto" sentinel.
	data := []byte(`{"id":"x","status":"completed","text":"hi","language_code":"en","speech_model":"universal-3-5-pro","audio_duration":1}`)
	r, err := parse(data, AutoModel)
	require.NoError(t, err)
	require.Equal(t, "universal-3-5-pro", r.Model)
}

func TestParse_AutoWithoutEchoKeepsSentinel(t *testing.T) {
	// Older responses omit speech_model; the sentinel is the honest fallback.
	data := []byte(`{"id":"x","status":"completed","text":"hi","language_code":"en","audio_duration":1}`)
	r, err := parse(data, AutoModel)
	require.NoError(t, err)
	require.Equal(t, AutoModel, r.Model)
}

func TestParse_PinnedModelIgnoresEcho(t *testing.T) {
	// When the user pinned a model, the requested value is what we report —
	// the echo can carry legacy labels like "best".
	data := []byte(`{"id":"x","status":"completed","text":"hi","language_code":"en","speech_model":"best","audio_duration":1}`)
	r, err := parse(data, "universal-2")
	require.NoError(t, err)
	require.Equal(t, "universal-2", r.Model)
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./internal/adapters/api/assemblyai/ -run 'TestParse_Auto|TestParse_PinnedModelIgnoresEcho' -v`

Expected: FAIL — `TestParse_AutoRecordsActualModel` reports `"auto" != "universal-3-5-pro"`. The other two pass already; they are regression guards.

- [ ] **Step 3: Write the implementation**

In `internal/adapters/api/assemblyai/parse.go`, add the field to `transcriptResponse`:

```go
type transcriptResponse struct {
	ID            string  `json:"id"`
	Status        string  `json:"status"`
	Error         string  `json:"error"`
	Text          string  `json:"text"`
	LanguageCode  string  `json:"language_code"`
	SpeechModel   string  `json:"speech_model"` // model actually used; echoed by the API
	AudioDuration float64 `json:"audio_duration"` // seconds
	Words         []aWord `json:"words"`
}
```

Then in `parse`, between the error check and the `res := &domain.Result{...}` literal, insert:

```go
	// In auto mode we never named a model, so report the one AssemblyAI ran.
	// The echo can carry legacy labels (older responses say "best"); we record
	// it verbatim as information and never feed it back into a request.
	if model == AutoModel && resp.SpeechModel != "" {
		model = resp.SpeechModel
	}
```

The `res := &domain.Result{...}` literal already reads `Model: model`, so it needs no change.

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test ./internal/adapters/api/assemblyai/ -v`

Expected: PASS for the whole package.

- [ ] **Step 5: Commit**

```bash
git add internal/adapters/api/assemblyai/parse.go internal/adapters/api/assemblyai/parse_test.go
git commit -m "feat(assemblyai): record the model the API actually used in auto mode"
```

---

### Task 5: Documentation and user-facing strings

The last references to the retired model live in help text and docs. Historical entries in `CHANGELOG.md` and files under `docs/plans/` and `.claude/plans/` are records of past work and must **not** be rewritten.

**Files:**
- Modify: `internal/delivery/gui/mainwindow.go:196`
- Modify: `README.md:156`
- Modify: `ROADMAP.md:160`
- Modify: `CHANGELOG.md` (new entry at the top of the unreleased section)

**Interfaces:**
- Consumes: the final model list from Task 1. No code depends on this task.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Update the GUI placeholder**

In `internal/delivery/gui/mainwindow.go` line 196, change:

```go
	m.speechModels.SetPlaceHolder("e.g. universal-3-pro,universal-2 — tried in order, falls back on language mismatch (assemblyai)")
```

to:

```go
	m.speechModels.SetPlaceHolder("leave empty for auto — or e.g. universal-3-5-pro,universal-2, tried in order (assemblyai)")
```

- [ ] **Step 2: Update the README example**

In `README.md` line 156, change:

```
transcribe sample.mp3 --api assemblyai --model universal-3-pro
```

to:

```
transcribe sample.mp3 --api assemblyai --model universal-3-5-pro
```

Directly below that command, add:

```markdown
AssemblyAI defaults to `auto`, which omits the model from the request so
AssemblyAI always applies its current best model (`universal-3-5-pro`, falling
back to `universal-2` for languages it does not cover). Pin a model only when
you need a specific one. The retired `universal-3-pro` is accepted and
silently upgraded to `universal-3-5-pro`, with a warning — AssemblyAI stops
accepting it on 2026-09-02.
```

- [ ] **Step 3: Update the ROADMAP flag reference**

In `ROADMAP.md` line 160, change:

```
- `--speech-models` (string, AssemblyAI). Fallback model list, e.g. `universal-3-pro,universal-2`. Pass through.
```

to:

```
- `--speech-models` (string, AssemblyAI). Fallback model list, e.g. `universal-3-5-pro,universal-2`. Pass through. Empty means auto (the key is omitted).
```

- [ ] **Step 4: Add the CHANGELOG entry**

At the top of the unreleased section of `CHANGELOG.md`, under a `### Changed` heading (create it if the section has none), add:

```markdown
- **AssemblyAI defaults to automatic model selection.** The default model is now
  `auto`, which omits `speech_models` from the request so AssemblyAI applies its
  current best model — `universal-3-5-pro`, falling back to `universal-2` for
  languages outside its 18. Pinning a specific model still works and still gets
  `universal-2` appended as a safety net.
- **`universal-3-pro` is retired.** AssemblyAI stops accepting it on 2026-09-02.
  It is no longer offered in the model list; existing configs, batch templates,
  and `--speech-models` values naming it are automatically upgraded to
  `universal-3-5-pro` with a warning.
- AssemblyAI results now record the model the API actually used, rather than
  `auto`, in the sidecar JSON.
```

- [ ] **Step 5: Verify no stale references remain in live files**

Run:

```bash
grep -rn "universal-3-pro" --include="*.go" --include="README.md" --include="ROADMAP.md" .
```

Expected: matches only in `internal/adapters/api/assemblyai/models.go` (the `deprecatedModels` map and the explanatory comments), `client_test.go` (the remap tests), and `models_test.go` (the "must not be offered" assertions). No match in `README.md`, `ROADMAP.md`, or any GUI/CLI string.

- [ ] **Step 6: Run the full test suite**

Run: `go test ./internal/adapters/... ./internal/core/... ./internal/delivery/cli/ ./internal/delivery/tui/`

Expected: PASS. Then attempt the full suite with `go test ./...`; if it fails only with a cgo/gcc error from the Fyne GUI package, build via `scripts/build.ps1` instead and note that the GUI change is a single string literal covered by compilation.

- [ ] **Step 7: Commit**

```bash
git add README.md ROADMAP.md CHANGELOG.md internal/delivery/gui/mainwindow.go
git commit -m "docs(assemblyai): document auto model selection and the universal-3-pro retirement"
```

---

## Manual verification

After all tasks, with a real AssemblyAI key configured:

1. `transcribe sample.mp3 --api assemblyai` — succeeds, and the sidecar JSON's model field names a concrete model (not `auto`).
2. `transcribe sample.mp3 --api assemblyai --speech-models universal-3-pro` — succeeds and prints the deprecation warning naming 2026-09-02.
3. GUI: select AssemblyAI; the model dropdown shows `auto` selected first, and the SRT/word-SRT/DaVinci checkboxes are all enabled.
