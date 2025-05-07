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

## In Progress Tasks

- [ ] Refactor to a unified API class
  - [ ] Create base transcription class with common methods
  - [ ] Implement AssemblyAI-specific implementation (submit and wait)
  - [ ] Implement direct response APIs (Groq, ElevenLabs)
  - [ ] Add error handling and retry logic

## Future Tasks

- [ ] Implement standardized parsers for each JSON format
  - [ ] Create parser for AssemblyAI format
  - [ ] Create parser for ElevenLabs format
  - [ ] Create parser for Groq format
  - [ ] Create unified data model for consistent access

- [ ] Unify all transcription scripts
  - [ ] Create master script with engine/model selection
  - [ ] Implement clicker interface similar to notion-cli
  - [ ] Add configuration management
  - [ ] Create unified command line interface

- [ ] Add back OpenAI official Whisper support
- [ ] Add local-whisper/faster-whisper as local transcription option

- [ ] Streamline output formatting options
  - [ ] Implement standard SRT output format
  - [ ] Implement word-level SRT output format
  - [ ] Implement DaVinci SRT output format
  - [ ] Implement plain text output format
  - [ ] Refactor timing options (fps, padding) to be more intuitive
  - [ ] Add format-specific configuration options

## Implementation Plan

The tool will be refactored to use a unified class architecture that handles all API interactions while providing a consistent interface. A common parser will handle different JSON formats from various APIs, transforming them into a standardized internal format.

Output options will be streamlined with sensible defaults while maintaining flexibility for different use cases.

### Relevant Files

- `transcribe.py` - Main entry point (to be created)
- `assemblyai_transcribe.py` - AssemblyAI implementation
- `elevenlabs_transcribe.py` - ElevenLabs implementation
- `groq_transcribe.py` - Groq implementation
- `utils/parsers.py` - JSON parsing utilities (to be created)
- `utils/formatters.py` - Output formatting utilities (to be created)