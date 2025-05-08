# Transcription Tool Implementation

A unified tool for transcribing audio using various APIs (AssemblyAI, ElevenLabs, Groq, OpenAI) with options for different output formats.

## Completed Tasks

- [x] AssemblyAI: Turn empty time frames between end of one word and start of next word into spacings
- [x] Add an explicit option `--show-pauses` that adds "(...)" text when pauses occur (default for `--davinci-srt`)

- [x] Create standardized word format helper function
  - [x] Implement `standardize_word_format()` in text_processing.py
  - [x] Convert AssemblyAI format (ms timestamps) to unified format (seconds)
  - [x] Ensure consistent spacing elements between words
  - [x] Add initial spacing for words not starting at timestamp 0
  - [x] Handle pause indicators based on silence threshold
  - [x] Update both scripts (elevenlabs & assemblyai) to use new helper function

- [x] Fix pause indicators in AssemblyAI SRT output
  - [x] Investigate why standardized format doesn't display pause indicators in SRT
  - [x] Create custom_export_subtitles function that uses standardized word format
  - [x] Update export_subtitles wrapper to use custom function when silence indicators needed
  - [x] Ensure consistent SRT creation with pause markers for both APIs

- [x] Have ElevenLabs and Groq use loguru like in AssemblyAI
- [x] Make sure all scripts skip re-encoding if a JSON is already present
- [x] Add the API that was used to the end of the JSON filename

- [x] Implement standardized parsers for each JSON format
  - [x] Create parser for AssemblyAI format
  - [x] Create parser for ElevenLabs format
  - [x] Create parser for Groq format
  - [x] Create unified data model for consistent access

- [x] Refactor to a unified API class
  - [x] Create base transcription class with common methods
  - [x] Implement AssemblyAI-specific implementation (submit and wait)
  - [x] Implement direct response APIs (Groq, ElevenLabs)
  - [x] Add error handling and retry logic; debug output and info with loguru

- [x] Unify all transcription scripts into a NEW central script
  - [x] Create master script with engine/model selection
  - [x] **CLI Framework:** Using Click instead of argparse for better user experience
  - [x] Created utils package with parsers, formatters, and API classes
  - [x] api (+ model where appropriate) selection via cli. defaults to groq
  - [x] check if API key is set and works before trying to access the selected api for more error robustness
  - [x] Create unified command line interface

- [x] pyinstaller and executable: preparation and setup
  - [x] Created package structure for Python package deployment
  - [x] Setup entry points for command-line usage
  - [x] Added PyInstaller configuration and build script
  
- [x] Add back OpenAI official Whisper support based on audio_transcribe.py
- [x] .env management for new users in user dir

## In Progress Tasks


## Future Tasks

- [ ] take a json file directly as input as well

- [ ] Add local-whisper/faster-whisper as local transcription option

- [ ] Streamline output formatting options
  - [ ] Implement standard SRT output format
  - [ ] Implement word-level SRT output format
  - [ ] Implement DaVinci SRT output format
  - [ ] Implement plain text output format
  - [ ] Refactor timing options (fps, padding) to be more intuitive
  - [ ] Add format-specific configuration options

- [ ] clean up. remove old unused scripts and util libaries

- [ ] double check language codes work as expected in all apis and if necessary write a converter (ie some api need de for German some deu or de_DE)

## Implementation Plan

The tool has been refactored to use a unified class architecture that handles all API interactions while providing a consistent interface. A common parser handles different JSON formats from various APIs, transforming them into a standardized internal format.

Output options have been streamlined with sensible defaults while maintaining flexibility for different use cases.

### Relevant Files

- ✅ `transcribe.py` - Main entry point (implemented)
- ✅ `assemblyai_transcribe.py` - AssemblyAI implementation (existing)
- ✅ `elevenlabs_transcribe.py` - ElevenLabs implementation (existing)
- ✅ `groq_transcribe.py` - Groq implementation (existing)
- ✅ `utils/parsers.py` - JSON parsing utilities (implemented)
- ✅ `utils/formatters.py` - Output formatting utilities (implemented)
- ✅ `utils/transcription_api.py` - Unified API classes (implemented)
- ✅ `utils/__init__.py` - Utils package initialization (implemented)

### How to Use the Unified Tool

The unified tool supports a consistent interface for all APIs:

```bash
# Basic usage
uv run .\transcribe.py --api assemblyai "path/to/audio.wav"

# With language selection
uv run .\transcribe.py --api groq --language de "path/to/audio.wav"

# With custom output formats
uv run .\transcribe.py --api elevenlabs --output text --output srt --output davinci_srt "path/to/audio.wav"

# With DaVinci Resolve optimized output
uv run .\transcribe.py --api groq --davinci-srt "path/to/audio.wav"
```

All APIs support the same command-line options, with sensible defaults for each API. The tool automatically detects existing transcripts and can regenerate different output formats from existing JSON files.