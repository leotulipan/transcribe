# Plan: ElevenLabs STT filter, AssemblyAI fallback models, GUI polish, About/version, affiliate links, README rewrite

## Context

After v0.10.0 reached CLI/GUI feature parity, six concrete issues surfaced during real-world use:

1. **ElevenLabs discover-models returns TTS models, not STT.** The `GET /v1/models` endpoint returns voice/TTS models (e.g. `eleven_v3`, `eleven_multilingual_v2`), not the `scribe_*` STT models. When the GUI user clicks "↻ Refresh Models" for ElevenLabs, the dropdown is populated with unusable IDs and transcription fails with HTTP 400 `unsupported_model`. The code in `internal/adapters/api/elevenlabs/discover.go:41-50` already has a code comment acknowledging this and explicitly defers the filter.
2. **AssemblyAI's modern model surface is hidden.** The hardcoded registry only exposes `best` / `nano`; AssemblyAI's current docs recommend an array like `["universal-3-pro", "universal-2"]` with automatic fallback. The plumbing for the array already exists (`SpeechModels` in `ProviderOpts`, `body["speech_models"]` in `client.go:196-198`), but the dropdown lags behind.
3. **GUI ergonomics.** Action buttons live only at the bottom (lost under scroll), Settings never auto-opens for first-run users with no keys, and the window icon (top-left of the Fyne window) is the default Fyne icon instead of the project's `assets/icon.ico` (only the tray/taskbar icon is set, via `cmd/transcribe-gui/resource.syso`).
4. **No About / version dialog.** `cmd/transcribe-gui/main.go:20` defines `var version = "dev"` and then `_ = version` (line 47) discards it. Users have no in-app way to confirm the build version or find the homepage.
5. **No affiliate / get-free-credits links in Settings.** The README has `https://dub.link/elevenlabs` (line 179) but the Settings dialog where users actually paste keys offers no prompts. The user has affiliate links for some providers and wants a registered list of which still need sign-up.
6. **README is outdated and developer-heavy.** It still references the removed Python tree (`audio_transcribe/`, `uv sync`, `pyproject.toml`, `--setup` wizard), only lists 4 of the 6 providers (missing Gemini, Mistral), never mentions the Windows installer or the GUI, and interleaves dev instructions with end-user quick-start.

Outcome: a focused PR series that fixes the immediate ElevenLabs breakage, surfaces AssemblyAI's better models with documented fallback semantics, ships a more polished GUI (top toolbar, auto-Settings, correct window icon, About dialog), adds revenue-relevant affiliate links to Settings, and rewrites the README for installer users.

## Approach

All changes are in the Go tree (`internal/`, `cmd/transcribe-gui/`, repo root). No Python work — that tree is gone. Tests are in `internal/.../*_test.go` next to the files they cover; add table-driven cases for each behavioral change.

### 1. ElevenLabs: filter discovery to `scribe_*` prefix

**File:** `internal/adapters/api/elevenlabs/discover.go:41-52`

Change the JSON struct to also pull `name` (useful for the warning case) and, more importantly, filter the result list. Keep only IDs with the `scribe_` prefix; if zero pass the filter, return the hardcoded `Models()` from `models.go:45-47` as a fail-safe so the user is never stranded with an empty dropdown.

```go
ids := make([]string, 0, len(payload))
for _, m := range payload {
    if strings.HasPrefix(m.ModelID, "scribe_") {
        ids = append(ids, m.ModelID)
    }
}
if len(ids) == 0 {
    ids = Models()
}
return discover.SortUnique(ids), nil
```

Update the doc-comment at lines 14-19 to reflect the new behavior (was: "we return every model_id"; becomes: "we filter to scribe_* — the only STT family today"). Add a unit test in `discover_test.go` (create if missing — there's no existing test file for this discover.go) with a fixture body containing mixed TTS + STT IDs, asserting only `scribe_*` survives.

### 2. AssemblyAI: expand model registry, document fallback

**File:** `internal/adapters/api/assemblyai/models.go:11-46`

Add capability entries for the modern Universal/SLAM family and update the priority order in `Models()`:

```go
// Best→worst order. Newer Universal models are English/EU-only;
// older "best" / "nano" cover everything else for safety.
func Models() []string {
    return []string{
        "universal-3-pro",
        "universal-3",
        "universal-2",
        "slam-1",
        "best",
        "nano",
    }
}
func DefaultModel() string { return "universal-3-pro" }
```

Each new key gets a `modelCaps` entry mirroring the existing "best" capability set (same accepted codecs, all four boolean flags). Keep `best` and `nano` so users with existing configs and older recordings still see them.

**Document fallback in the GUI:**
- `internal/delivery/gui/mainwindow.go:178-179` — change the placeholder on `m.speechModels` from `"model1,model2 (assemblyai fallbacks)"` to `"e.g. universal-3-pro,universal-2 — tried in order, falls back on lang mismatch"`.
- Add a tooltip via `widget.NewLabel` next to the Provider hints accordion section header (or simply update the help text inline) noting that AssemblyAI's `speech_models` array is honored only when set; otherwise the dropdown model is used alone.

Tests: in `internal/adapters/api/assemblyai/client_test.go` (existing — already covers `speech_models` array per the explore), add a case that the new `universal-3-pro` model name is accepted by `submitTranscript` and posted under the `speech_model` JSON key. No new endpoint logic needed; the plumbing is already correct.

### 3. GUI polish — top toolbar, auto-Settings, window icon

**File:** `internal/delivery/gui/mainwindow.go`

#### 3a. Top fixed toolbar (keeping bottom row)

Refactor the layout in `newMainWindow` (`mainwindow.go:252-279`). Today the entire form is wrapped in a `container.NewScroll`, which means buttons scroll out of view. Switch to a Border layout so the toolbar is pinned:

```go
topBar := container.NewHBox(m.startBtn, m.cancelBtn, settingsBtn, aboutBtn)
scroll := container.NewScroll(container.NewPadded(form))
w.SetContent(container.NewBorder(topBar, nil, nil, nil, scroll))
```

`form` keeps its existing bottom-row `container.NewHBox(m.startBtn, m.cancelBtn, settingsBtn)` — Fyne happily renders the same widget reference in two parents; click handlers fire identically. (Confirmed by Fyne's widget identity model; no double-fire risk for stateless buttons.)

Add an `aboutBtn` (`widget.NewButton("About…", m.onAbout)`) shown only in the top bar to avoid double-clutter at the bottom.

#### 3b. Auto-open Settings on launch when zero keys exist

Add a helper near `preferredProvider` (`mainwindow.go:291-301`):

```go
func hasAnyAPIKey(cfg ports.Config) bool {
    for _, v := range cfg.APIKeys {
        if strings.TrimSpace(v) != "" {
            return true
        }
    }
    return false
}
```

In `gui.Run` (or wherever `newMainWindow` is wrapped — see `internal/delivery/gui/app.go` or equivalent; explore noted `gui.Run` is the entry), after `w.Show()`, check `if !hasAnyAPIKey(deps.Config()) { m.onSettings() }`. The existing `onSettings` (`mainwindow.go:685-688`) already wires save → refreshProviders, so the user can immediately Start once they save.

#### 3c. Window icon

The window icon (top-left of the Fyne window chrome) is *not* the same as the OS taskbar icon (which `resource.syso` controls). Fyne needs an explicit `fyne.Resource` set via `w.SetIcon(...)` or `a.SetIcon(...)`.

**Add** a small file `internal/delivery/gui/icon.go`:

```go
package gui

import _ "embed"
import "fyne.io/fyne/v2"

//go:embed assets/icon.png
var iconPNG []byte

var appIcon = fyne.NewStaticResource("transcribe.png", iconPNG)
```

The embedded resource has to live under `internal/delivery/gui/assets/` because `go:embed` only sees files under the package's directory. Copy `assets/icon-1024.png` (the 17 KB PNG already in repo at `assets/icon-1024.png`) into `internal/delivery/gui/assets/icon.png` as part of this change. PNG works for cross-platform Fyne; `.ico` does not.

Call `a.SetIcon(appIcon)` in `gui.Run` once on the `fyne.App`, before any `NewWindow`. Fyne propagates the app icon to new windows automatically; no per-window call needed.

### 4. About / version dialog

**File:** `internal/delivery/gui/about.go` (new) + `cmd/transcribe-gui/main.go`.

#### 4a. Surface the version

`cmd/transcribe-gui/main.go:20` declares `var version = "dev"`. Drop the `_ = version` and instead pass it into the deps:

- Extend `gui.Deps` (in `internal/delivery/gui/deps.go`, wherever `NewDeps` is defined) with a `Version string` field.
- Update `gui.NewDeps(...)` signature OR — simpler — add a separate `NewDepsWithVersion` constructor to avoid churning callers; pick whichever the existing code style prefers. (Likely just extend `NewDeps` since main.go is the only caller.)
- In `main.go`, wire `gui.NewDeps(svc, cfg, log, saveCfg, loadCfg, buildSvc, version)`. Ensure `cmd/transcribe-gui/build_version.go` (or the build pipeline) injects the real version at link time via `-ldflags="-X main.version=v0.11.0"`. Check whether `installer/` or `build.py` already does this; if not, add a line to the build script. The existing `installer/transcribe.iss:40` (`transcribe-setup-v{#AppVersion}.exe`) implies the version is already known at build time, so the ldflags injection should pair with that.

#### 4b. About dialog

In `internal/delivery/gui/about.go`:

```go
func (m *mainWindow) onAbout() {
    content := container.NewVBox(
        widget.NewLabelWithStyle("Audio Transcribe", fyne.TextAlignCenter, fyne.TextStyle{Bold: true}),
        widget.NewLabel("Version " + m.deps.Version()),
        widget.NewLabel("by Leonard Tulipan"),
        widget.NewHyperlink("leotulipan.at", parseURL("https://leotulipan.at")),
        widget.NewHyperlink("github.com/leotulipan/transcribe", parseURL("https://github.com/leotulipan/transcribe")),
        widget.NewLabel("MIT License"),
    )
    dialog.ShowCustom("About", "Close", content, m.Window)
}
```

`parseURL` is a tiny helper wrapping `url.Parse`; either put it in `about.go` or reuse if a similar helper already exists.

### 5. Affiliate links in Settings

**File:** `internal/delivery/gui/settings.go:99-108`

Replace the simple `FormItem` rows with an HBox-per-row that pairs each entry with a clickable hyperlink labeled "Get key" or "Register / get free credits". Use `widget.NewHyperlink`. The link target for each provider:

| Provider | Affiliate / referral link | Notes |
|---|---|---|
| ElevenLabs | `https://dub.link/elevenlabs` | confirmed by user, already in README |
| AssemblyAI | `https://www.assemblyai.com/` (placeholder) | user wants to know if there's an affiliate; **leave as plain link, note in PR description** |
| Groq | `https://console.groq.com/keys` | no known affiliate |
| OpenAI | `https://platform.openai.com/settings/organization/api-keys` | no free credits |
| Gemini | `https://aistudio.google.com/apikey` | free tier exists; no affiliate |
| Mistral | `https://console.mistral.ai/api-keys` | free tier; no affiliate |

Layout sketch — replace each `widget.NewFormItem("ElevenLabs", eleven)` with a wrapper that puts the entry on the left and the "Get key" hyperlink on the right:

```go
elevenRow := container.NewBorder(nil, nil, nil,
    widget.NewHyperlink("Get key (free credits)", parseURL("https://dub.link/elevenlabs")),
    eleven,
)
form := widget.NewForm(
    widget.NewFormItem("ElevenLabs", elevenRow),
    // ...repeat per provider
)
```

After implementation, **the PR description should include a "provider affiliate links — please register for any missing" checklist** for the user. Candidates known to operate referral programs as of writing: ElevenLabs (dub.link), AssemblyAI (Impact-based), Groq (none public), OpenAI (none), Gemini (none), Mistral (none). The user can register and replace URLs once accepted.

### 6. README rewrite

**File:** `README.md` (full rewrite, replacing all 345 current lines).

Structure for the new README:

1. **Hero + 30-second pitch** — what it does, supported providers (all six), what makes it different (DaVinci Resolve auto-cut, word-level SRT).
2. **Quick Start (Windows installer)**
   - Download `transcribe-setup-vX.Y.Z.exe` from Releases.
   - Run installer.
   - Launch "Transcribe" from Start Menu → Settings auto-opens → paste API keys → Save.
   - Drop a file on the GUI, click Start.
   - Right-click an audio/video file → "Transcribe with..." context menu.
3. **Getting API keys** — same provider list as today but with affiliate / free-credits notes, expanded to include Gemini and Mistral. Mirror the Settings dialog so users see the same options.
4. **GUI walkthrough** (with one or two screenshots if available; otherwise textual description). Mention Advanced accordion, output formats, DaVinci Resolve options.
5. **CLI quick-reference** — installed `transcribe.exe` syntax, common flags. Keep brief; full list via `transcribe.exe --help`.
6. **DaVinci Resolve workflow** — pause markers, filler-words, auto-cut, frame-accurate timing. This is a unique selling point; keep most of the current §"DaVinci Resolve Optimized" text.
7. **Output formats** — txt, srt, word.srt, davinci.srt.
8. **Troubleshooting** — short list.
9. **Privacy / where keys are stored** — Windows `%LOCALAPPDATA%\transcribe\` (verify exact path via `configadapter.New().Path()` referenced in `settings.go:50`).
10. **For developers** (now at the end, condensed) — Go toolchain, `go build ./cmd/transcribe`, `go build ./cmd/transcribe-gui`, `go test ./...`, installer build (`installer/transcribe.iss`), Go project layout pointer to `docs/engineering-guide.md`. Drop all Python references and the `uv sync` / `.python-version` / `audio_transcribe/` content.
11. **Contributing / License / Support / Changelog** — keep concise.

Cross-link the new README to `leotulipan.at` in the footer.

## Critical files

- `internal/adapters/api/elevenlabs/discover.go` — filter discovery to `scribe_*`.
- `internal/adapters/api/elevenlabs/models.go` — update the doc comment if needed; no logic change.
- `internal/adapters/api/assemblyai/models.go` — expand registry; change default.
- `internal/adapters/api/assemblyai/client_test.go` — table case for new model.
- `internal/delivery/gui/mainwindow.go` — top toolbar, auto-Settings hook, About button.
- `internal/delivery/gui/settings.go` — affiliate hyperlinks per provider.
- `internal/delivery/gui/about.go` *(new)* — About dialog.
- `internal/delivery/gui/icon.go` *(new)* — embedded PNG icon resource.
- `internal/delivery/gui/assets/icon.png` *(new)* — copy of `assets/icon-1024.png`.
- `internal/delivery/gui/deps.go` (or equivalent) — add `Version` field.
- `cmd/transcribe-gui/main.go` — drop `_ = version`, pass `version` into `NewDeps`, confirm `-ldflags -X main.version=...` injection.
- `README.md` — full rewrite.

## Verification

End-to-end manual + automated checks. After each behavioral change, run `go test ./internal/adapters/api/...` to make sure nothing regressed.

1. **ElevenLabs filter** — run `go test ./internal/adapters/api/elevenlabs/...`. Then build the GUI (`go build ./cmd/transcribe-gui`), launch it, select ElevenLabs, click "↻". Expect dropdown to contain only `scribe_*` entries.
2. **AssemblyAI models** — GUI: switch to AssemblyAI; dropdown should now show `universal-3-pro` first. CLI: `transcribe.exe sample.mp3 --api assemblyai --model universal-3-pro` should succeed against a real key. Try the fallback path in the GUI by entering `universal-3-pro,universal-2` in the Provider hints CSV and submitting a non-English file — should not 4xx.
3. **GUI top toolbar** — launch, expand all Advanced accordion sections, scroll. Confirm Start / Cancel / Settings / About stay visible at the top.
4. **Auto-Settings** — delete the config file (path printed in Settings dialog), relaunch. Settings should open immediately on top of the main window. Paste any key, save, confirm dialog closes and Start works.
5. **Window icon** — launch the built `transcribe-gui.exe`. Inspect the top-left of the window: should show the Audio Transcribe icon, not the Fyne default. Compare with the (already-correct) Windows taskbar icon.
6. **About dialog** — click "About…". Verify version string matches the built binary (`transcribe-gui.exe --version` if added, otherwise the installer filename). Click the `leotulipan.at` hyperlink — browser opens.
7. **Affiliate links in Settings** — open Settings; each provider row has a hyperlink. Click each one and confirm the correct destination opens.
8. **README** — preview rendered Markdown locally (`gh markdown-preview README.md` or any preview tool). Confirm zero Python references, all six providers covered, installer flow up top, dev section at the bottom.

When all eight pass, the work is complete and ready to land as either one PR or a small series (`fix(elevenlabs)`, `feat(assemblyai)`, `feat(gui)`, `docs(readme)`).
