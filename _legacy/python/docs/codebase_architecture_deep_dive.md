# Audio Transcribe - Deep Codebase Research Report

## Executive Summary

**Audio Transcribe** is a sophisticated Python-based CLI tool for transcribing audio and video files using multiple AI transcription APIs (AssemblyAI, ElevenLabs, Groq, OpenAI, Google Gemini, Mistral Voxtral). The project emphasizes:

- **Multi-API provider pattern** with abstract base classes
- **PyAV-accelerated audio processing** (2-3x faster than subprocess)
- **Standalone Windows executable distribution** via PyInstaller
- **DaVinci Resolve-optimized SRT output** with auto-cut markers
- **Interactive TUI setup wizard** using questionary/rich
- **Smart audio optimization** cascade (passthrough → extract → FLAC → MP3)
- **Chunking support** for files exceeding API limits

**Version**: 0.4.5
**Framework**: Click (not Typer)
**Build Backend**: hatchling (migrated from setuptools)
**Python**: 3.9+ (tested on 3.11/3.13)

---

## Architecture Overview

```
audio_transcribe/
├── cli.py                          # Click CLI entry point (main dispatcher)
├── utils/
│   ├── api/                        # Provider pattern implementations
│   │   ├── base.py                 # TranscriptionAPI abstract base class
│   │   ├── chunking.py             # ChunkingMixin for large files
│   │   ├── assemblyai.py
│   │   ├── elevenlabs.py
│   │   ├── groq.py
│   │   ├── openai.py
│   │   ├── openai_extended.py      # Extended features
│   │   ├── gemini.py
│   │   └── mistral_voxtral.py
│   ├── parsers.py                  # TranscriptionResult + API-specific parsers
│   ├── formatters.py               # Output format generators (SRT/text/JSON)
│   ├── config.py                   # ConfigManager for .env/settings
│   ├── models.py                   # MODEL_REGISTRY for API models
│   ├── adapters.py                 # Parameter normalization
│   ├── defaults.py                 # DefaultsManager for CLI defaults
│   ├── ffmpeg.py                   # FFmpeg detection utilities
│   └── __init__.py                 # get_api_instance() factory
├── transcribe_helpers/
│   ├── audio_processing.py         # PyAV/ffmpeg audio operations
│   ├── pyav_backend.py             # PyAV implementations (fast path)
│   ├── output_formatters.py        # Low-level SRT/text generation
│   ├── text_processing.py          # Filler words, pause detection
│   ├── language_utils.py           # Language code validation
│   ├── chunking.py                 # File chunking logic
│   ├── intermediate_files.py      # IntermediateFileManager
│   └── utils.py                    # Logging setup
└── tui/
    ├── wizard.py                   # Interactive setup wizard
    └── interactive.py              # Interactive mode (run without args)
```

---

## Core Design Patterns

### 1. Provider Pattern (API Implementations)

All transcription APIs inherit from `TranscriptionAPI` abstract base class:

```python
class TranscriptionAPI(ABC):
    # Capability flags - override in subclasses
    supports_word_timestamps: bool = True
    supports_segment_timestamps: bool = False
    supports_speaker_diarization: bool = False
    supports_srt_format: bool = True
    supported_output_formats: List[str] = ["text", "json"]

    @abstractmethod
    def transcribe(self, audio_path, **kwargs) -> TranscriptionResult:
        pass

    @abstractmethod
    def check_api_key(self) -> bool:
        pass
```

**Concrete implementations**:
- `GroqAPI(TranscriptionAPI, ChunkingMixin)` - Inherits chunking capability
- `OpenAIAPI(TranscriptionAPI, ChunkingMixin)` - Inherits chunking capability
- `AssemblyAIAPI(TranscriptionAPI)` - Direct upload, no chunking needed (200MB limit)
- `ElevenLabsAPI(TranscriptionAPI)` - Direct upload (1000MB limit)

**Factory function** (`utils/__init__.py`):
```python
def get_api_instance(api_name: str, api_key: str = None) -> TranscriptionAPI:
    """Factory function to get API instance by name."""
    api_classes = {
        "assemblyai": AssemblyAIAPI,
        "elevenlabs": ElevenLabsAPI,
        "groq": GroqAPI,
        "openai": OpenAIAPI,
        "gemini": GeminiAPI,
        "mistral": MistralVoxtralAPI,
    }
    return api_classes[api_name](api_key=api_key)
```

### 2. Chunking Mixin Pattern

For APIs with strict file size limits (Groq 25MB, OpenAI 25MB):

```python
class ChunkingMixin:
    """Mixin for APIs that support chunking large files."""

    def transcribe_with_chunking(self, audio_path, chunk_length, overlap, **kwargs):
        # Split audio into chunks
        # Transcribe each chunk
        # Merge results with timestamp adjustment
```

The mixin is used by Groq and OpenAI APIs to handle long audio files.

### 3. Standardized Result Format

`TranscriptionResult` in `parsers.py` provides unified data structure:

```python
@dataclass
class TranscriptionResult:
    text: str                          # Full transcript text
    confidence: float                  # 0.0-1.0
    language: str                      # ISO-639-1/3 code
    words: List[Dict[str, Any]]        # Word-level timestamps
    speakers: List[Dict[str, Any]]     # Speaker info (if diarization)
    segments: List[Dict[str, Any]]     # Segment-level info
    api_name: str                      # Source API
```

**Word entry structure**:
```python
{
    'text': 'hello',           # Word text
    'start': 1.234,            # Start time (seconds or ms)
    'end': 1.567,              # End time
    'type': 'word',            # 'word' | 'spacing' | 'audio_event'
    'confidence': 0.98,        # Optional
    'speaker_id': 'A',         # Optional
}
```

**Spacing elements** separate words and can represent pauses:
```python
{
    'text': ' ',              # or ' (...) ' for marked pauses
    'start': 1.567,
    'end': 1.678,
    'type': 'spacing',
}
```

### 4. PyAV Backend with FFmpeg Fallback

Audio processing prioritizes PyAV (2-3x faster) with automatic fallback to ffmpeg subprocess:

```python
# In pyav_backend.py
def get_duration_seconds(audio_path: Path) -> float:
    """Get duration 10x faster than ffprobe."""
    if PYAV_AVAILABLE:
        try:
            with av.open(str(audio_path)) as container:
                return container.duration / 1_000_000
        except Exception:
            pass
    return _get_duration_ffprobe(audio_path)  # Subprocess fallback
```

**Audio optimization cascade** (`audio_processing.py`):

1. **Passthrough check** - If file < threshold (100MB) and format compatible → skip processing
2. **Audio extraction** (if video) - PyAV stream copy with ffmpeg fallback
3. **FLAC conversion** (if API requires) - 16kHz mono for Groq/OpenAI
4. **MP3 compression** (last resort) - 128k mono if still exceeds limit

**`OptimizationResult`** metadata:
```python
@dataclass
class OptimizationResult:
    path: Path                         # Optimized file path
    is_temporary: bool                 # Whether to cleanup after use
    size_mb: float                     # File size in MB
    bytes_per_second: float            # For chunking calculations
    intermediate_manager: Optional[IntermediateFileManager]
```

### 5. Intermediate File Manager

Deferred cleanup pattern for intermediate files:

```python
class IntermediateFileManager:
    """Manages intermediate files with consistent naming and deferred cleanup."""

    def get_path_for(self, operation: FileOperation, extension: str) -> Path:
        """Generate consistent path for intermediate file."""
        # e.g., input.mp4 → input_intermediate_extracted.m4a

    def cleanup(self) -> None:
        """Clean up all registered intermediate files."""
```

This allows reusing intermediate files and ensures cleanup even on errors.

---

## Audio Processing Pipeline

### Input Handling Flow

```
cli.py::process_file()
    ↓
    ↓ Check if JSON exists (reuse mode)
    ↓ check_json_exists() → load_json_data() → parse_*_format()
    ↓
    ↓ If no JSON, optimize audio
    ↓ optimize_audio_for_api()
    ↓     → can_passthrough() # Skip processing if small
    ↓     → extract_audio_pyav() / extract_audio_from_mp4()
    ↓     → convert_to_flac_pyav() / convert_to_flac()
    ↓     → convert_to_mp3_pyav() / convert_to_mp3()
    ↓
    ↓ Transcribe via API
    ↓ api_instance.transcribe()
    ↓     → ChunkingMixin.transcribe_with_chunking() (if needed)
    ↓     → Parse response to TranscriptionResult
    ↓
    ↓ Generate outputs
    ↓ create_output_files()
    ↓     → process_filler_words() (optional)
    ↓     → standardize_word_format() (add spacing elements)
    ↓     → create_srt() / create_text_file()
```

### API-Specific Format Requirements

```python
API_FORMAT_REQUIREMENTS = {
    "groq": {"requires_flac": True, "accepts_video": False},
    "openai": {"requires_flac": True, "accepts_video": False},
    "assemblyai": {"requires_flac": False, "accepts_video": True},
    "elevenlabs": {"requires_flac": False, "accepts_video": True},
    "gemini": {"requires_flac": False, "accepts_video": True},
    "mistral": {"requires_flac": False, "accepts_video": True},
}
```

### File Size Limits by API

| API | Limit | Chunking Support |
|-----|-------|------------------|
| AssemblyAI | 200MB | No (direct upload) |
| ElevenLabs | 1000MB | No (direct upload) |
| Groq | 25MB | Yes (ChunkingMixin) |
| OpenAI | 25MB | Yes (ChunkingMixin) |
| Gemini | Limited | No |
| Mistral | Limited | No |

---

## Output Formatting

### SRT Format Types

**Three SRT modes** generated by `create_srt()` in `output_formatters.py`:

1. **standard**: Sentence-level subtitles with line wrapping
2. **word**: Each word as its own subtitle (for precise editing)
3. **davinci**: DaVinci Resolve optimized with:
   - Pause markers `(...)` for auto-cut
   - Filler words as separate UPPERCASE lines
   - Frame-accurate timing offsets

### DaVinci Resolve Features

The `--davinci-srt` flag creates SRT files optimized for DaVinci Resolve Studio:

```python
# Pause detection
if show_pauses and gap > silence_threshold:
    pause_text = " (...) "  # DaVinci recognizes this as auto-cut point

# Filler word handling
if filler_lines:
    # Output as separate UPPERCASE line
    subtitle_line = word['text'].upper()
```

**Key options**:
- `--silent-portions 350`: Mark pauses >350ms as `(...)`
- `--filler-lines`: Output filler words as UPPERCASE subtitle lines
- `--padding-start -125`: Start 125ms earlier (frame accuracy)
- `--padding-end 0`: No end padding

### Text Processing Functions

**`standardize_word_format()`** - Adds spacing elements between words:
```python
# Input: Basic words list without spacing
words = [
    {'text': 'hello', 'start': 0, 'end': 0.5},
    {'text': 'world', 'start': 0.5, 'end': 1.0}
]

# Output: With spacing elements
standardized = [
    {'text': 'hello', 'start': 0, 'end': 0.5, 'type': 'word'},
    {'text': ' ', 'start': 0.5, 'end': 0.5, 'type': 'spacing'},
    {'text': 'world', 'start': 0.5, 'end': 1.0, 'type': 'word'}
]
```

**`process_filler_words()`** - Removes fillers and merges with surrounding pauses:
```python
# Detects filler words: "äh", "ähm", "um", "uh"
# Merges: [prev_spacing] + [filler] + [next_spacing] → [merged_pause]
```

---

## Configuration Management

### ConfigManager

**Locations**:
- Windows: `%LOCALAPPDATA%\audio_transcribe\.env`
- Linux/Mac: `~/.audio_transcribe/.env`
- Fallback: CWD `.env` (for local override, read-only)

**Storage**:
- API keys: `.env` file via `python-dotenv`
- User preferences: `config.json`

```python
class ConfigManager:
    def set_api_key(self, api_name: str, key: str) -> None:
        env_var = f"{api_name.upper()}_API_KEY"
        os.environ[env_var] = key
        set_key(self.env_file, env_var, key)
```

### DefaultsManager

Effective parameters = CLI args + config file + preset defaults:

```python
kwargs = DefaultsManager.get_effective_params(
    api_name="groq",
    raw_user_params={"language": "de"},
    preset="davinci"  # Applies DaVinci-specific defaults
)
```

**Preset system**:
- `"davinci"`: `silent_portions=350`, `padding_start=-125`, `remove_fillers=True`
- Default: Standard transcription settings

---

## CLI Implementation (Click)

### Entry Point

```python
@click.group(invoke_without_command=True)
@click.argument("input_path", required=False, type=NormalizedPath(exists=False))
@click.option("--api", "-a", help="API to use")
@click.option("--language", "-l", help="Language code")
@click.option("--output", "-o", multiple=True, help="Output formats")
def main(ctx, input_path, api, language, output, ...):
    """Unified Audio Transcription Tool."""
```

**Special features**:
- `NormalizedPath` Click type: Strips trailing quotes/backslashes (PowerShell issue)
- Auto-detects JSON input: Enables `--use-json-input` when file ends with `.json`
- Interactive mode: Runs `run_interactive_mode()` when no arguments provided

### Key CLI Options

| Option | Description |
|--------|-------------|
| `--api`, `-a` | API to use (assemblyai, elevenlabs, groq, openai, gemini, mistral) |
| `--language`, `-l` | ISO-639-1/3 code (e.g., en, de, fr) |
| `--output`, `-o` | text, srt, word_srt, davinci_srt, json, all |
| `--davinci-srt`, `-D` | DaVinci Resolve optimized output |
| `--silent-portions`, `-p` | Mark pauses >X ms with (...) |
| `--filler-lines` | Output fillers as UPPERCASE subtitle lines |
| `--remove-fillers` | Remove filler words |
| `--word-srt`, `-C` | Each word as separate subtitle |
| `--setup` | Run interactive setup wizard |
| `--use-json-input`, `-j` | Accept JSON files as input |
| `--force`, `-r` | Re-transcribe even if transcript exists |

---

## TUI (Text User Interface)

### Setup Wizard (`tui/wizard.py`)

Uses **questionary** and **rich** for an interactive setup experience:

```python
def run_setup_wizard():
    """Interactive configuration of API keys and defaults."""
    # Main menu:
    # - Configure assemblyai (Not Configured)
    # - Configure elevenlabs (Configured)
    # - Configure groq (Not Configured)
    # - Configure openai (Not Configured)
    # - Configure Defaults
    # - Exit
```

**Features**:
- Live API key validation via `check_api_key()`
- Masked key display
- Default settings: API, language, output formats

### Interactive Mode (`tui/interactive.py`)

Launched when running CLI without arguments:
- File/folder selection
- API selection
- Language selection
- Output format selection
- DaVinci options

---

## Build & Distribution

### PyInstaller Build (`build.py`)

**Build process**:
1. Install dependencies via UV
2. Run PyInstaller with hidden imports
3. Create zip archive with executable + batch templates

**PyInstaller configuration**:
```python
run_command([
    "pyinstaller",
    f"--name=transcribe-windows-amd64",
    "--onefile",
    "--noconfirm",
    "--clean",
    f"--add-data={loguru_path};loguru",  # Windows separator
    "--hidden-import=assemblyai",
    "--hidden-import=groq",
    "--hidden-import=openai",
    # ... more hidden imports
    cli_path
])
```

**Output**:
- `dist/transcribe-windows-amd64.exe` (standalone executable)
- `dist/transcribe-windows-amd64.zip` (release archive)

### Dependency Management (UV + Hatchling)

**pyproject.toml**:
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "audio-transcribe"
version = "0.4.5"
requires-python = ">=3.9"
dependencies = [
    "av>=12.0.0",              # PyAV for fast audio
    "click",
    "loguru",
    "pydub-ng",                # Migrated from pydub (unmaintained)
    "python-dotenv",
    "requests",
    "assemblyai",
    "groq",
    "openai>=1.78.0",
    "google-generativeai>=0.8.0",
    "mistralai>=1.0.0",
    "questionary>=2.1.0",
    "rich>=14.2.0",
]

[project.scripts]
transcribe = "audio_transcribe.cli:main"
```

---

## Key Implementation Details

### JSON Reuse Mode

Automatic detection of existing transcripts:
1. Check for `filename_apiname.json` (API-specific)
2. Check for `filename.json` (generic)
3. If found, parse and reuse (skip API call)

```python
api_json_path = file_dir / f"{file_name}_{api_name}.json"
generic_json_path = file_dir / f"{file_name}.json"

if api_json_path.exists():
    return True, api_json_path
```

### macOS Metadata Filtering

Skips macOS resource fork files during folder processing:

```python
files = [f for f in files if not f.name.startswith('.')]
# Filters: .DS_Store, ._filename, etc.
```

### Pydub-ng Migration

Migrated from unmaintained `pydub` to `pydub-ng` (v0.4.5):

```python
# In pyproject.toml
dependencies = [
    "pydub-ng",  # Was: "pydub"
]
```

### FFmpeg Detection (Windows)

**Detection order**:
1. System PATH
2. `FFMPEG_PATH` environment variable
3. WinGet installation
4. Program Files
5. Chocolatey
6. Scoop
7. Bundled fallback

Results cached to avoid repeated filesystem lookups.

### PyAV 16.x Compatibility Fix

**Issue**: PyAV 16.x changed `add_stream()` API.

**Fix** (commit bc53739):
```python
# Old API (PyAV <16):
out_stream = out.add_stream('flac', rate=16000)

# New API (PyAV 16+):
out_stream = out.add_stream(template=audio_stream)
# Or:
out_stream = out.add_stream('flac', rate=16000)
```

---

## Testing Approach

**Manual testing** with fixtures in `tests/fixtures/`:
- Audio files: `sample_speech.wav`, `sample_multi_speaker.wav`, etc.
- Expected outputs: `sample_speech.txt`, `sample_speech.json`, etc.
- API-specific JSON: `sample_speech_groq.json`, `sample_speech_assemblyai.json`

**Test structure**:
```
tests/
├── fixtures/
│   ├── audio_files/          # Input audio samples
│   └── expected_outputs/     # Expected transcription results
├── unit/                     # Unit tests
├── integration/              # API integration tests
└── acceptance/               # End-to-end tests
```

**No automated test runs**: Tests are run manually with `uv run script.py test/fixtures/sample.wav`.

---

## Known Gotchas & Edge Cases

### 1. PowerShell Path Escaping
```python
class NormalizedPath(click.Path):
    def convert(self, value, param, ctx):
        if value:
            value = value.rstrip('"\'').rstrip()  # Strip trailing quotes
            value = value.rstrip('/\\')           # Strip trailing slashes
        return super().convert(value, param, ctx)
```

### 2. Scientific Notation in Timestamps
AssemblyAI sometimes returns timestamps like `2.4e-07`:

```python
# In parsers.py
if isinstance(start, (int, float)):
    start_str = str(start)
    if 'e' in start_str.lower() or abs(start) < 0.0001 and start != 0:
        start = 0  # Fix scientific notation
```

### 3. Groq Timestamp Format
Groq uses **seconds with decimal precision** (0.5s intervals):

```python
# Detection in standardize_word_format()
if (isinstance(start_val, float) and start_val != int(start_val)):
    is_decimal_format = True  # Groq format
```

### 4. Mistral Language Constraint
Mistral Voxtral auto-detects language; manual language is ignored:

```python
if api_name == "mistral" and kwargs.get("language"):
    logger.warning("Mistral Voxtral auto-detects language. "
                   "Requested language will be ignored.")
```

### 5. Word Timestamp Unavailability
Some APIs don't provide word timestamps:

```python
if api_instance.supports_word_timestamps is False:
    logger.warning(f"{api_name} does not provide word-level timestamps. "
                   f"SRT files will use approximate timing estimates.")
```

---

## Batch File Templates

Located in `batch_templates/` for drag-and-drop workflows:

**`transcribe_elevenlabs_de.bat`**:
```bat
@echo off
transcribe.exe "%~1" --api elevenlabs --language de --davinci-srt --silent-portions 350 --padding-start -125
```

**`transcribe_groq_de.bat`**:
```bat
@echo off
transcribe.exe "%~1" --api groq --language de --output srt,text
```

Users can drag-and-drop audio files onto these `.bat` files for one-click transcription.

---

## Dependency Notes

| Dependency | Version | Purpose |
|------------|---------|---------|
| `click` | Latest | CLI framework |
| `loguru` | Latest | Logging |
| `pydub-ng` | 0.2.0+ | Audio processing (migrated from pydub) |
| `av` | 12.0.0+ | PyAV for fast audio (2-3x faster) |
| `python-dotenv` | Latest | .env file management |
| `requests` | Latest | HTTP client |
| `assemblyai` | Latest | AssemblyAI client |
| `groq` | Latest | Groq client |
| `openai` | 1.78.0+ | OpenAI client |
| `google-generativeai` | 0.8.0+ | Gemini client |
| `mistralai` | 1.0.0+ | Mistral client |
| `questionary` | 2.1.0+ | Interactive prompts |
| `rich` | 14.2.0+ | Rich terminal output |

---

## Future Improvements (from 2026-01-23 Plan)

1. **Streaming chunking** - Reduce memory usage for large files
2. **Enhanced IntermediateFileManager** - Consistent naming + deferred cleanup
3. **CLI `--size-threshold` option** - User-configurable passthrough threshold
4. **FFmpeg subprocess caching** - Avoid repeated filesystem lookups
5. **Refactored `optimize_audio_for_api()`** - Cleaner passthrough logic

---

## Conclusion

**Audio Transcribe** is a well-architected CLI tool that demonstrates:

- **Clean separation of concerns** via provider pattern and mix-ins
- **Performance optimization** through PyAV backend (2-3x faster)
- **User-friendly features** (TUI setup, batch files, drag-and-drop)
- **Professional output** (DaVinci Resolve integration, multiple SRT formats)
- **Robust error handling** (fallback mechanisms, retry logic)
- **Cross-platform support** (Windows-focused, but Unix-compatible)

The codebase is production-ready with comprehensive documentation, testing fixtures, and a working build pipeline for standalone executables.
