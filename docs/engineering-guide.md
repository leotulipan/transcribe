# Transcribe — Engineering Guide

This guide provides detailed guidance for working with code in this repository. The
repository `CLAUDE.md` links here for all code work.

## Project Overview

**Audio Transcribe** is a unified multi-API transcription tool supporting AssemblyAI, ElevenLabs, Groq, and OpenAI. It provides both CLI and TUI (Text User Interface) for users to transcribe audio/video files with multiple output formats including plain text, standard SRT, word-level SRT, and DaVinci Resolve-optimized SRT with auto-cut markers for pause detection.

The tool is designed to work standalone without requiring Python installation (ships as a Windows x64 executable via PyInstaller).

## Common Development Tasks

### Running the CLI

```bash
# Install dependencies with UV
uv sync

# Run the main CLI
uv run transcribe.py --help

# Or via the package entry point
uv run python -m audio_transcribe.cli --help

# Examples
uv run transcribe.py "path/to/audio.mp3" --api groq --language de
uv run transcribe.py "path/to/video.mp4" --api elevenlabs --davinci-srt
```

### Building the Executable

```bash
# Build standalone Windows x64 executable (includes all dependencies)
python build.py

# Output: dist/transcribe-windows-amd64.exe + zip archive
```

### Environment Setup

Configuration is managed by `ConfigManager` (audio_transcribe/utils/config.py):
- Windows: `%LOCALAPPDATA%\audio_transcribe\.env`
- Linux/Mac: `~/.audio_transcribe/.env`

Running `uv run transcribe.py --setup` launches the TUI setup wizard which:
- Validates API keys interactively
- Stores credentials in the config directory (never committed)
- Loads from both the central `.env` and local `.env` (local overrides)

### Testing

The project uses manual testing with test files in `test/files/`. Test acceptance criteria are documented in `test/ACCEPTANCE_CRITERIA.md`.

```bash
# Run a test transcription
uv run transcribe.py test/files/sample.wav --api groq --output text,srt

# Test output goes to current directory or specified location
# Verify results manually or with acceptance criteria
```

## Architecture & Design Patterns

### Core Architecture

```
audio_transcribe/
├── cli.py                          # Click CLI entry point (main command dispatcher)
├── utils/
│   ├── api/                        # API implementations (provider pattern)
│   │   ├── base.py                 # TranscriptionAPI abstract base class
│   │   ├── assemblyai.py
│   │   ├── elevenlabs.py
│   │   ├── groq.py
│   │   ├── openai.py
│   │   └── chunking.py             # Mixin for handling large files
│   ├── parsers.py                  # TranscriptionResult standardized format
│   ├── formatters.py               # Output format generators (text/SRT/etc.)
│   ├── config.py                   # ConfigManager for settings/env vars
│   ├── models.py                   # MODEL_REGISTRY for API-specific models
│   ├── adapters.py                 # Parameter normalization across APIs
│   ├── defaults.py                 # DefaultsManager for CLI defaults
│   └── __init__.py                 # get_api_instance() factory function
├── transcribe_helpers/             # Transcription utilities
│   ├── audio_processing.py         # ffmpeg/pydub wrappers, compression, chunking
│   ├── output_formatters.py        # SRT/text/JSON format helpers
│   ├── text_processing.py          # Speaker labels, filler words, pause detection
│   ├── language_utils.py           # Language code validation/conversion
│   ├── chunking.py                 # File chunking logic
│   └── utils.py                    # General utilities (logging setup, etc.)
└── tui/                            # Interactive setup and wizards
    ├── interactive.py              # Interactive mode (run without arguments)
    ├── wizard.py                   # Setup wizard for API keys
    └── __init__.py                 # TUI exports
```

### Provider Pattern (Multiple APIs)

Each transcription API (AssemblyAI, ElevenLabs, Groq, OpenAI) is a concrete implementation of the abstract `TranscriptionAPI` base class (audio_transcribe/utils/api/base.py):

- **Base class** defines the contract: `transcribe()`, `check_api_key()`, `list_models()`
- **Concrete implementations** inherit and implement API-specific logic
- **Factory function** `get_api_instance()` in `utils/__init__.py` instantiates the correct provider based on `--api` flag

Example flow:
```
cli.py → get_api_instance(api_name="groq") → GroqAPI(api_key) → transcribe()
```

### Standardized Result Format

`TranscriptionResult` (audio_transcribe/utils/parsers.py) provides a unified data structure:
- Fields: `text`, `confidence`, `language`, `words` (word-level timestamps), `speakers`, `segments`
- Parsers convert API-specific JSON responses to this format (parser functions like `parse_assemblyai_response()`)
- Output formatters consume `TranscriptionResult` and generate SRT, text, JSON, etc.

### Chunking Mixin for Large Files

`ChunkingMixin` in audio_transcribe/utils/api/chunking.py handles files exceeding API size limits:
- Splits audio into chunks, transcribes each, merges results
- Used by APIs with strict size limits (e.g., Groq 25MB, OpenAI 25MB)
- Automatically invoked in audio processing pipelines

### SRT Output Variants

Multiple SRT format types are handled by `create_srt_file()` in formatters.py:
- **standard**: Standard SRT with sentence-level subtitles
- **word_srt**: Word-level subtitles (each word = one subtitle)
- **davinci_srt**: DaVinci Resolve optimized with:
  - Pause markers as `(...)` for auto-cut
  - Filler words as separate UPPERCASE lines
  - Customizable timing offsets (--padding-start, --silent-portions)

## Key Implementation Details

### CLI Entry Point (cli.py)

Main command dispatcher with Click. Key responsibilities:
- Argument/option parsing
- File path validation and expansion
- Check for existing transcripts (skip re-processing)
- Check for existing JSON (reuse parsed data with `use_json_input`)
- Folder recursion (find all audio/video files)
- Call transcription pipeline
- Generate output files

**Note**: The `--file` and `--folder` options are deprecated in favor of positional arguments.

### Audio Processing Pipeline

`audio_transcribe/transcribe_helpers/audio_processing.py`:
1. **Format check**: Validate file format (audio or video)
2. **Compression**: Convert to optimal format for API (FLAC, PCM, MP3 based on API)
3. **Size checking**: Verify against API limits, compress or chunk if needed
4. **Extraction**: Extract audio from video files (MP4, MOV, etc.)
5. **Optimization**: Balance quality/filesize for the specific API

Key functions: `check_audio_length()`, `check_file_size()`, `optimize_audio_for_api()`

### Model Registry

`MODEL_REGISTRY` in models.py defines available models per API with defaults. Extend this when new models are released by different providers.

### Configuration Management

`ConfigManager` in config.py:
- Loads from central user directory (Windows LOCALAPPDATA, Unix ~/.audio_transcribe)
- Stores `.env` file with API keys
- Loads local `.env` as override (for development)
- Provides typed accessors for settings

### Speaker Labels & Diarization

ElevenLabs supports speaker diarization (--diarize flag). When enabled:
- Parser extracts speaker info from response
- SRT formatters can optionally include speaker labels (--speaker-labels)
- Speaker info stored in `TranscriptionResult.speakers`

### Filler Words & Pause Detection

`text_processing.py`:
- Default filler words: "um", "uh", "ähm", "äh", "hm", "hmm"
- Customizable via --filler-words flag
- For DaVinci SRT: marks pauses longer than --silent-portions threshold as `(...)`
- `--remove-fillers` strips them from output
- `--filler-lines` outputs them as separate UPPERCASE subtitle lines

### Language Handling

`language_utils.py` validates and converts language codes:
- Accepts ISO-639-1 (e.g., "de", "en") and ISO-639-3 (e.g., "deu", "eng")
- Normalizes to lowercase for API compatibility
- Different APIs expect different code formats

## Testing

Uses **pytest** with Red/Green TDD workflow. Full guide: `python/docs/red_green_tdd.md`

```bash
# Unit tests (fast, no API keys needed)
uv run python -m pytest tests/unit/ -v

# Integration tests (needs API keys + ffmpeg)
uv run python -m pytest tests/integration/ -v

# All tests, stop on first failure
uv run python -m pytest tests/ -v -x

# Run by marker
uv run python -m pytest tests/ -m unit
uv run python -m pytest tests/ -m integration
```

**Structure**: `tests/unit/` (fast, no externals), `tests/integration/` (API keys, ffmpeg), `tests/fixtures/` (sample audio + expected outputs), `tests/conftest.py` (shared fixtures, LLM judge).

**Markers**: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`, `@pytest.mark.requires_ffmpeg`

Integration tests auto-skip when API keys are missing or invalid. Tests requiring ffmpeg auto-skip when it's not on PATH.

For DaVinci SRT validation, import generated `.srt` into DaVinci Resolve and verify pause markers, filler word formatting, and timing accuracy.

## Building & Distribution

### Build Process (build.py)

The build.py script:
1. Installs dependencies via UV (including PyInstaller)
2. Runs PyInstaller with hidden imports (loguru, assemblyai, groq, etc.)
3. Bundles the executable into dist/
4. Creates a zip archive for distribution

Hidden imports are necessary because PyInstaller can't detect dynamic imports in some libraries (e.g., loguru's dynamic module loading).

### Executable Distribution

- Single file: `transcribe-windows-amd64.exe`
- No Python installation required
- Includes all dependencies and assets
- Batch files in `batch_templates/` can be paired with the executable for drag-and-drop workflows

### Versioning

Semantic versioning in pyproject.toml (Major.Minor.Patch). Update version before release.

## Important Implementation Notes

### API Key Validation

Always call `check_api_key()` on the API instance before transcribing. This catches missing/invalid credentials early and provides useful error messages.

### Retry Logic

Transcription APIs sometimes timeout or rate-limit. `TranscriptionAPI` base class includes:
- `max_retries = 3`
- `retry_delay = 5` seconds

Implement retry loops in concrete API classes. Example:
```python
def transcribe(self, audio_path, **kwargs):
    for attempt in range(self.max_retries):
        try:
            return self._transcribe_internal(audio_path, **kwargs)
        except (TimeoutError, RateLimitError) as e:
            if attempt < self.max_retries - 1:
                logger.warning(f"Retry {attempt + 1}/{self.max_retries}")
                time.sleep(self.retry_delay)
            else:
                raise
```

### Output File Organization

By default, output files are created next to the input file:
- Input: `/path/to/audio.mp3`
- Output: `/path/to/audio.txt`, `/path/to/audio.srt`, etc.

With `--output-dir`: output goes to specified directory, preserving base filename.

### JSON Input Mode (use_json_input)

When a JSON file exists for an input file (e.g., `audio_groq.json` or `audio.json`), the parser automatically loads and uses it instead of re-transcribing. This enables:
- Reusing transcriptions with different output formats
- Testing output formatters without API calls

Auto-detection happens in cli.py via `check_json_exists()`.

### Logging

Uses `loguru` configured in `transcribe_helpers/utils.py`:
- Default level: INFO
- `--verbose` flag: DEBUG level
- `--debug` flag: DEBUG with additional internal logging
- Logs go to stderr (rich colors preserved)

## Shell Commands

**Never prefix Bash commands with `cd`** unless you genuinely need to run a
command in a directory *other* than the current working directory. The session's
cwd is set at start and persists across every Bash invocation — `cd <our-cwd>
&& cmd` is identical to `cmd`, just slower and noisier.

This applies in every situation the habit shows up:

- `cd <repo-root> && git status` → just `git status`. Git already operates on the worktree.
- `cd <repo-root> && go build ./...` → just `go build ./...`.
- `cd <worktree-path> && grep -rn foo` → just `grep -rn foo`.
- `cd "<path with spaces>" && cmd` → never; spaces trigger a "may execute hooks
  from target directory" approval prompt.

**Worktree note:** When this session runs inside a git worktree (e.g.
`.claude/worktrees/<branch>/`), that worktree path *is* your cwd. Don't `cd`
into it — you're already there.

**Subagent note:** Subagents inherit the dispatcher's cwd. If you're dispatching
one, do *not* instruct it to `cd` anywhere either; trust the cwd. When a
subagent genuinely needs a different directory, name it in the prompt and have
the subagent use `git -C <path>`, `go -C <path>`, or absolute paths instead of
`cd && ...`.

The only legitimate use of `cd` is when a single sequenced command needs to
start somewhere genuinely else (e.g. `cd /tmp/scratch && make-something && cp
result ./`). Prefer absolute paths or per-command `-C`/`--cwd` flags over `cd`
even there.

**Wrong:** `cd "C:\local\cc\dev\python\transcribe" && go test ./...`
**Right:** `go test ./...`

**Wrong:** `cd .claude/worktrees/go-port && git log`
**Right:** `git -C .claude/worktrees/go-port log` (only when actually targeting a
different dir)

Each `cd` to a path containing spaces costs the user a manual approval click on
Windows. Don't make them click.

## Git Workflow

- Main branch: production releases
- Keep CHANGELOG.md updated with all changes
- Atomic commits with semantic meaning (feat:, fix:, refactor:, etc.)
- Use pull requests for significant changes

## Dependencies

**Core CLI**: Click, loguru, python-dotenv, requests, pydub
**APIs**: assemblyai, groq, openai
**TUI**: questionary, rich
**Build**: pyinstaller (dev-only)

Managed via `pyproject.toml` with UV (never pip directly).

## When Extending the Project

### Adding a New Transcription API

1. Create `audio_transcribe/utils/api/new_api.py`
2. Inherit from `TranscriptionAPI` base class
3. Implement `transcribe()`, `check_api_key()`, `list_models()`
4. Add parser function in `parsers.py` if API response format is unique
5. Register in `get_api_instance()` factory function
6. Update MODEL_REGISTRY with available models
7. Update CLI help text and README
8. Add unit tests in `tests/unit/test_parsers.py` and integration tests in `tests/integration/test_api_integration_newapi.py` (see `python/docs/red_green_tdd.md`)

### Adding a New Output Format

1. Add formatter function to `formatters.py` (e.g., `create_custom_format_file()`)
2. Update `create_output_files()` to handle the new format
3. Add option to CLI in cli.py (e.g., `--custom-format`)
4. Add tests in `tests/unit/test_formatters.py` and CLI test in `tests/integration/test_cli.py`

### API Limit or Model Changes

- Update `get_api_file_size_limit()` in audio_processing.py for size limits
- Update MODEL_REGISTRY in models.py for available models (single source of truth)
- Update README.md with new feature description
- Run `uv run python -m pytest tests/ -v` to verify parsers still work
