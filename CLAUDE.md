# CLAUDE.md

## Identity

**Audio Transcribe** is a unified multi-API transcription tool — a CLI and TUI for
transcribing audio and video with AssemblyAI, ElevenLabs, Groq, OpenAI, Gemini, and
Mistral, producing plain-text, SRT, word-level SRT, and DaVinci-Resolve SRT output. It
ships as a standalone Windows executable, so no Python install is required to run it.
The project is open-source (github.com/leotulipan/transcribe, MIT licensed). A Go port
is in progress alongside the original Python package.

## Resources

| Resource | Read when... |
|---|---|
| `docs/engineering-guide.md` | **...writing or changing any code here — read this first.** Architecture, the provider pattern, testing, build, shell rules, and extension steps. |
| `README.md` | ...you need user-facing usage or feature information |
| `ROADMAP.md`, `CHANGELOG.md` | ...planning work or preparing a release |
| `.claude/plans/` | ...picking up an existing implementation plan |

## Workflow

1. For any code work, read `docs/engineering-guide.md` first.
2. Python: use `uv` (`uv sync`, `uv run`). The Go port: use `go` in `cmd/` and `internal/`.
3. Develop test-first; keep `CHANGELOG.md` updated; make atomic, semantic commits.
4. Build the standalone Windows executable with `python build.py`.

## Engineering Conventions

The full conventions — shell rules, the provider pattern, the standardized
`TranscriptionResult` format, retry logic, testing markers, and extension steps — live
in `docs/engineering-guide.md`. Defer to it for any code change. Keep this repository
public-clean: no personal or business information in tracked files.
