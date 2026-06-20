# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.0] - 2026-06-20

### Added
- macOS builds: official Apple Silicon (`arm64`) and Intel (`amd64`)
  binaries are now published on the Releases page alongside the Windows
  installer and portable zip.
- CI/release: GitHub Actions now build and test on Windows + macOS for every
  push, and a `v*` tag triggers a full multi-platform release (installer,
  zip, macOS tarballs, SHA256 sidecars, auto-generated notes). See
  `docs/release/`.
- Diarization: speaker IDs from providers are normalized so ElevenLabs'
  `speaker_0`/`speaker_1` render as `[Speaker 0]`/`[Speaker 1]` instead of
  `[Speaker speaker_0]`. AssemblyAI's `A`/`B` pass through unchanged.
- Diarization: plain-text output now emits speaker labels (one paragraph per
  speaker turn, prefixed `[Speaker X]:`) when `--speaker-labels`/`--diarize`
  is set.
- `merge` subcommand: transcribe two or more single-speaker tracks (e.g. one
  mic per podcast participant) separately — each gets its own SRT/JSON/text —
  then interleave them by timestamp into a combined SRT + text labeled with
  each speaker's name. Tracks are assigned via repeatable
  `--speaker LABEL=FILE`; `--offset LABEL=DURATION` aligns tracks that don't
  share an exact zero point. Named labels render as `[Julia]:` rather than
  `[Speaker X]:`.

- GUI: top toolbar (Start / Cancel / Settings / About) pinned outside the
  scroll area so primary actions stay reachable when the Advanced accordion
  is expanded; bottom button row kept for muscle memory.
- GUI: About dialog with build version, leotulipan.at, GitHub link, and
  license. Version is injected via `-ldflags "-X main.version=…"`.
- GUI: window-chrome icon (top-left of the window) now uses the project
  icon instead of Fyne's default. Embedded as a PNG via `go:embed`.
- GUI: Settings dialog auto-opens on launch when no API keys are
  configured, so first-run users land directly on the key form.
- GUI: per-provider "Get key" hyperlinks in Settings — ElevenLabs uses
  the dub.link affiliate URL; the rest point at each provider's canonical
  key page (most have a free tier or free credits).
- AssemblyAI: model registry expanded to surface the Universal / SLAM
  family (`universal-3-pro`, `universal-3`, `universal-2`, `slam-1`)
  alongside the legacy `best` / `nano`. Default changed to
  `universal-3-pro`. The existing `--speech-models csv` flag still
  passes an ordered fallback array per AssemblyAI's docs.
- Dev: `scripts/install-local.ps1` (and `scripts/build.ps1 -Install`) build,
  copy the binaries into the local install locations, and verify the version
  and `merge` subcommand in one command.

### Changed
- ElevenLabs: dynamic model discovery (`GET /v1/models`) now filters to
  the `scribe_*` STT family. Previously the unfiltered list pulled in
  TTS models (`eleven_v3`, `eleven_multilingual_v2`, …) and any pick
  resulted in HTTP 400 `unsupported_model` at transcribe time. Falls back
  to the hardcoded list if zero `scribe_*` IDs are returned.
- README rewritten for installer users: Quick start lives at the top
  (download → run installer → drop file → Start), all six providers are
  listed with key-page links, Python tooling references removed, and the
  developer section is now a short appendix at the end.

### Fixed
- GUI/CLI: ffmpeg and ffprobe no longer flash a console window on Windows
  (child processes spawn with CREATE_NO_WINDOW). Combined with the GUI being
  built `-H windowsgui`, double-clicking the app opens no stray console.
- GUI: files sent from Windows now reach the app — dragging a file onto the
  desktop shortcut, or right-click → "Transcribe with…", pre-fills the picker
  (previously the path was ignored).
- GUI: the activity log is readable again — it was drawn in Fyne's muted
  "disabled" colour (grey-on-grey) and now uses full foreground contrast.
- Audio: WAV, M4A and MP4 inputs are sent to the provider as-is instead of
  being needlessly re-encoded. Provider "accepted formats" now match against
  the file's container, not just its codec.
- CLI/GUI: the "chunk 1/1" progress line is suppressed for single-chunk jobs
  (it appears only when a file is actually split).

## [0.10.0] - 2026-05-27

Feature-parity release. The Go CLI now matches the Python tool's flag surface
and the GUI exposes everything the CLI can do.

### Added
- CLI: short-flag aliases for parity with Python (`-a/-l/-o/-c/-C/-D/-m/-p`
  `/-w/-d/-v/-e/-r/-j/-J`). `-V` is the version flag; `-c` enables the new
  `--chars-per-line` wrapper.
- CLI: 28 new flags wiring the previously-only-Python behaviors —
  `--chars-per-line`, `--words-per-subtitle`, `--word-srt`, `--start-hour`,
  `--diarize`, `--speaker-labels`, `--num-speakers`, `--keyterms-prompt`,
  `--speech-models`, `--padding-start`, `--padding-end`, `--fps`,
  `--fps-offset-start`, `--fps-offset-end`, `--show-pauses`,
  `--silent-portion-ms`, `--silent-portions` (alias),
  `--filler-words`, `--remove-fillers`, `--filler-lines`,
  `--size-threshold`, `--chunk-length`, `--overlap`, `--use-input`,
  `--use-pcm`, `--keep`, `--keep-flac`,
  `--force`, `--save-cleaned-json`, `--use-json-input`, `--extensions`,
  `--list`, `--debug`, `--verbose`, `--api-key`.
- End-to-end diarization support for assemblyai + elevenlabs, including
  `[Speaker X]:` prefixes in SRT/DaVinci output.
- Word-level SRT output (one subtitle per word) for click-to-edit workflows.
- DaVinci subtitle padding (`--padding-start`/`--padding-end`) and frame-grid
  snapping (`--fps`, `--fps-offset-start`, `--fps-offset-end`).
- TUI interactive setup wizard for first-run API-key entry, plus an extended
  options flow with a language picker and advanced flags.
- GUI: collapsible Advanced section exposing the full CLI flag surface —
  subtitle wrapping, diarization, DaVinci timing, fillers, audio pipeline,
  I/O & workflow, and provider hints.
- `DefaultModel(provider)` on the service port so UIs can pre-select a
  sensible model without poking adapter internals.

### Changed
- Provider adapters now return their model list as a deterministic
  best→worst slice (was random map iteration). UIs surface the strongest
  option first.
- `Service.ListProviders()` is sorted in a canonical order (ElevenLabs,
  AssemblyAI, Groq, OpenAI, Gemini, Mistral) so the GUI dropdown is stable
  between runs.
- GUI defaults: provider dropdown picks ElevenLabs when configured;
  model dropdown initialises to `provider.DefaultModel()`.
- `mistral` default model: `voxtral-mini-latest` → `voxtral-small-latest`
  (higher tier).

### Fixed
- `--start-hour` is now honoured in word-level SRT output (previously it
  only applied to the standard SRT writer).
- DaVinci padding is applied before pause markers, not after, so pause gaps
  aren't accidentally clipped.

### Known follow-ups (deferred)
- Chunker overlap dedup: `--overlap` shifts chunk starts but the merger does
  not yet de-duplicate the overlapping words.
- `--silent-portions` always wins over `--silent-portion-ms` when both are
  set (cobra limitation; not true last-wins).
- TUI: pressing Esc on the advanced screen quits the app instead of
  returning to the format step.

## [0.9.0] - 2026-05-18

First proper release of the Go port. The Python tool (`audio_transcribe/`)
continues to live in the repo and ship its own 0.x line; this release covers
only the Go binary at `bin/transcribe.exe` + `bin/transcribe-gui.exe`.

### Added
- Go port: clean-architecture rewrite with ports/adapters/services layout.
- `transcribe test-keys [--json]` — validate every configured provider API
  key with a non-consuming endpoint (`GET /models` for most, `/v1/user` for
  ElevenLabs, `/v2/transcript?limit=1` for AssemblyAI). Reports
  ok / invalid / missing / unsupported / error per provider.
- `transcribe discover-models [--provider X] [--json] [--dry-run]` — fetch
  live model lists from groq/openai/mistral/gemini/elevenlabs and cache
  them under `[discovered_models]` in `config.toml`. Service prefers
  discovered lists over hardcoded fallbacks.
- GUI: folder picker + drag-and-drop for batch transcription (sequential).
- GUI: per-provider model refresh button (↻ next to the dropdown).
- GUI: settings dialog hot-reloads the service in-process — no restart
  needed after saving API keys, ffmpeg path, etc.
- Config: repo-local `.transcribe.toml` walk-up override (gitignored), in
  addition to user-level `%LOCALAPPDATA%\transcribe\config.toml`.
- Build script: PowerShell `scripts/build.ps1` produces both
  `transcribe.exe` (CLI/TUI/GUI/JSON modes) and `transcribe-gui.exe`
  (console-less GUI for taskbar pinning), with semver version embedded.
- Atomic TOML writes (write-temp + rename) to survive concurrent saves
  from the GUI and `discover-models`.

### Fixed
- ffmpeg path resolution: when the user pastes a shim directory (e.g.
  `%LOCALAPPDATA%\Microsoft\WinGet\Links\`) instead of the full executable
  path, `audio.New` now appends the binary name and validates a regular
  file before storing. CLI fails fast on invalid input; GUI falls back to
  PATH discovery with a warning. New `ResolveBinary` also checks WinGet,
  Chocolatey, and Scoop shim directories on Windows.

### Notes
- Python tool retained at repo root (`audio_transcribe/`, `tests/`,
  `pyproject.toml`); planned move to `python/` in a later cleanup pass.
- Python-only docs already moved to `python/docs/`, `python/plans/`.

## [0.2.0] - 2025-01-XX

### Added
- Public release preparation
- Batch templates directory with ready-to-use .bat files, icons, and shortcuts
- GitHub Actions workflows for automated CI/CD and releases
- Comprehensive documentation (CONTRIBUTING, CODE_OF_CONDUCT, SECURITY)
- GitHub issue and pull request templates
- Release checklist documentation
- Enhanced build script that creates zip archives with executables

### Changed
- Complete README rewrite focused on end-user experience
- Moved legacy documentation to separate `legacy-docs` branch
- Improved build process to include LICENSE and README in release artifacts

### Security
- API keys stored securely in user profile directory (not in project directory)
- Removed legacy .env files from repository

## [0.1.4] - Previous versions

### Added
- Initial public release
- Support for multiple transcription APIs (AssemblyAI, ElevenLabs, Groq, OpenAI)
- Multiple output formats (text, SRT, word-level SRT, DaVinci Resolve optimized)
- Interactive setup wizard for API key configuration
- Standalone Windows executable

### Changed
- Moved from individual API scripts to unified CLI tool

[Unreleased]: https://github.com/leotulipan/transcribe/compare/v0.11.0...HEAD
[0.11.0]: https://github.com/leotulipan/transcribe/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/leotulipan/transcribe/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/leotulipan/transcribe/compare/v0.2.0...v0.9.0
[0.2.0]: https://github.com/leotulipan/transcribe/compare/v0.1.4...v0.2.0

