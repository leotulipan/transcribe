# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/leotulipan/transcribe/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/leotulipan/transcribe/compare/v0.2.0...v0.9.0
[0.2.0]: https://github.com/leotulipan/transcribe/compare/v0.1.4...v0.2.0

