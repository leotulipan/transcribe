# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

## Testing Approach

No framework is used. Manual testing with test files:

1. Place test audio/video files in `test/files/`
2. Run CLI with test file and check output in current directory
3. Document test cases and expected results in `test/ACCEPTANCE_CRITERIA.md`
4. Output files are generated in the same directory as the input file (or specified with --output-dir)

When testing DaVinci features, import the generated `.srt` into DaVinci Resolve and verify:
- Pause markers are recognized as auto-cut points
- Filler words are highlighted in UPPERCASE
- Timing is frame-accurate

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
8. Test with manual test files

### Adding a New Output Format

1. Add formatter function to `formatters.py` (e.g., `create_custom_format_file()`)
2. Update `create_output_files()` to handle the new format
3. Add option to CLI in cli.py (e.g., `--custom-format`)
4. Test with sample TranscriptionResult

### API Limit or Model Changes

- Update `get_api_file_size_limit()` in audio_processing.py for size limits
- Update MODEL_REGISTRY in models.py for available models
- Update README.md with new feature description
- Test with real API responses to ensure parsers still work
