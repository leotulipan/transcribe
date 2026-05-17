# Python staging area

This directory holds Python-only artifacts that have been moved out of the
repo root as part of the Go port. The Python source code (`audio_transcribe/`,
`tests/`, `pyproject.toml`, etc.) is still at the repo root and will be
relocated into `python/` in a later cleanup pass.

## Layout

- `docs/` — Python codebase architecture, TDD workflow, release checklist.
- `plans/` — historical planning docs (`2026-01-23 Plan.md`, `features.md`,
  `project_outline.md`) and per-sprint walkthroughs under `plans/sprints/`.
- `web-assets/` — orphan front-end fragments (`i18n/`) from an earlier
  voice-recorder webapp idea. No current code references; kept for history.

The Go port lives at the repo root: `cmd/`, `internal/`, `go.mod`, etc.
