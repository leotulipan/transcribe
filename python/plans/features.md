# Implemented Features

This document tracks the chronological history of features implemented in the Audio Transcribe tool, detailing the user requests and the technical changes made to satisfy them.

## 1. Codebase Modularization & Refactoring
**Request:** Refactor the monolithic `transcribe.py` and `utils/transcription_api.py` into a clean, modular package structure to improve maintainability and extensibility.

**Implementation:**
The project was restructured into a proper Python package named `audio_transcribe`. The massive `transcription_api.py` was decomposed into individual files within `audio_transcribe/utils/api/`: `base.py` (base class), `groq.py`, `openai.py`, `elevenlabs.py`, and `assemblyai.py`.

`cli.py` was updated to serve as the main entry point, importing these modularized components. The root `transcribe.py` was retained as a thin wrapper for backward compatibility. This separation of concerns allows for easier addition of new APIs and features.

## 2. Interactive TUI (Text User Interface)
**Request:** Create a "really solid" TUI for setup and execution to improve the user experience, particularly for users who prefer not to use command-line arguments.

**Implementation:**
The `questionary` and `rich` libraries were integrated to build a modern CLI experience. `audio_transcribe/tui/wizard.py` was created to provide a guided setup process for configuring API keys. `audio_transcribe/tui/interactive.py` was implemented to offer an interactive workflow for file selection, API choice, and parameter configuration.

The CLI logic in `cli.py` was updated to automatically trigger this interactive mode when the tool is run without arguments, enabling a seamless "drag-and-drop" style usage.

## 3. Standardized Audio Processing
**Request:** Ensure consistent audio handling (extraction, conversion, file size limits) across all API implementations.

**Implementation:**
All API classes (`GroqAPI`, `OpenAIAPI`, etc.) were updated to utilize shared helper functions from `transcribe_helpers/audio_processing.py`. This includes `extract_audio_from_mp4` for handling video inputs and `get_api_file_size_limit` for enforcing API-specific constraints.

Logic was standardized to check file sizes before processing and automatically trigger chunking or format conversion (e.g., to FLAC for OpenAI) only when necessary, ensuring robust handling of various media types.

## 4. Dynamic API Model Discovery
**Request:** Enhance the tool to dynamically fetch available models from API providers instead of relying solely on hardcoded lists, and use this for API key validation.

**Implementation:**
The `TranscriptionAPI` base class in `audio_transcribe/utils/api/base.py` was updated to include an abstract `list_models` method. Concrete implementations were added for `GroqAPI`, `OpenAIAPI`, and others to query their respective endpoints (e.g., `GET /v1/models`).

The `check_api_key` methods were updated to use these `list_models` calls as a practical validation step, ensuring that the provided API key effectively has access to the service.

## 5. Build Pipeline for Windows x64
**Request:** Create a build pipeline that generates a self-contained Windows executable, specifically for the x64 platform.

**Implementation:**
`build.py` was updated to handle platform-specific naming, generating artifacts like `transcribe-windows-amd64.exe`. The script automates the installation of dependencies (including `uv` and `pyinstaller`) and the build process.

A GitHub Actions workflow (`.github/workflows/release.yml`) was created to automatically build and release this executable upon tagging a new version, ensuring a reproducible and automated release process.

## 6. TUI & CLI Polish
**Request:** Refine the TUI setup wizard, improve API selection visuals, and add versioning.

**Implementation:**
`audio_transcribe/tui/interactive.py` was updated to visually indicate configured vs. unconfigured APIs, preventing users from selecting unusable options. A rich progress bar was added to `cli.py` to provide better feedback during batch processing.

`audio_transcribe/tui/wizard.py` was refined to use clearer status indicators. Additionally, a `--version` flag was added to the CLI via `click.version_option` to allow users to easily check the installed version.

## 7. Centralized Configuration & Defaults
**Request:** Centralize configuration and environment variables to avoid scattered `.env` files and allow users to save default preferences.

**Implementation:**
`ConfigManager` in `audio_transcribe/utils/config.py` was updated to store configuration and the `.env` file in a platform-specific user directory (e.g., `%LOCALAPPDATA%/audio_transcribe` on Windows). It now supports loading a local `.env` file from the current working directory as an override.

The TUI (`wizard.py`) was enhanced with a "Configure Defaults" menu, allowing users to set their preferred API, language, and output formats. These defaults are persisted in `config.json` and automatically applied in interactive mode (`interactive.py`) and as fallbacks in the CLI.

## 8. Audio Optimization & Positional Arguments
**Request:** Support positional arguments for input files/folders and implement smart audio conversion to handle API file size limits.

**Implementation:**
`audio_transcribe/cli.py` was updated to accept the input file or folder as a positional argument (e.g., `transcribe my_audio.mp3`), simplifying usage. The legacy `--file` and `--folder` flags remain as optional alternatives.

`audio_transcribe/transcribe_helpers/audio_processing.py` was enhanced with an `optimize_audio_for_api` function. It implements a cascade of strategies to reduce file size:
1.  Check if the original file fits the API limit.
2.  If it's a video, extract the audio track (no re-encoding).
3.  Convert to FLAC (lossless compression).
4.  Convert to MP3 128kbps mono (lossy compression).

A `--keep` flag was added to the CLI to optionally preserve these optimized intermediate files; otherwise, they are automatically cleaned up after processing.

## 9. TUI Color Fixes & Optimization Refinements
**Request:** Fix the display of color instructions in the setup wizard (green for Configured, red for Not Configured) and ensure audio optimization logic is not blocked by redundant API-level checks.

**Implementation:**
`audio_transcribe/tui/wizard.py` was updated to use standard ANSI escape codes instead of raw Rich tags, ensuring that "Configured" and "Not Configured" statuses are correctly colored in the terminal.

`audio_transcribe/utils/api/assemblyai.py` was refactored to remove redundant file size checks and audio extraction logic. The API class now relies entirely on the centralized `optimize_audio_for_api` workflow in `cli.py`, preventing premature "File too large" errors and allowing the optimization cascade to function correctly.

## 10. SRT Generation Bug Fix (start_hour=None TypeError)
**Request:** Fix the bug where SRT files contained only "1" instead of full subtitle content, and enable JSON file reuse to avoid wasting API credits.

**Problem:** When running the transcribe tool, SRT files were being created but only contained the first subtitle number ("1"). The error `TypeError: unsupported operand type(s) for +=: 'int' and 'NoneType'` was occurring during timestamp formatting.

**Root Cause:** The `--start-hour` CLI option had `default=None`, which was being passed through kwargs to the SRT generation functions. When the code attempted `hours += start_hour`, it failed because you cannot add `None` to an integer.

**Implementation:**
- **`audio_transcribe/cli.py` (line 454)**: Changed `--start-hour` option default from `None` to `0`
- **`audio_transcribe/cli.py` (line 455)**: Updated version from `0.1.1` to `0.1.2`
- **`audio_transcribe/utils/formatters.py` (line 95)**: Changed `kwargs.get("start_hour", 0)` to `kwargs.get("start_hour") or 0` to properly handle `None` values
- **`audio_transcribe/transcribe_helpers/output_formatters.py` (lines 70, 89)**: Added defensive None handling with `hours += (start_hour or 0)` pattern

**Status:** Completed. The installation issue was resolved by reinstalling the tool. The SRT generation fix was verified by processing an existing JSON file, which correctly produced the full SRT output. The JSON reuse functionality works as expected.

**Known Issue:** The CLI argument parsing behavior requires options (like `--api`) to be placed before positional arguments when invoking the main command directly, or relies on interactive mode defaults. This is a minor usability quirk but does not block functionality.

## 11. Cleanup & Organization
**Request:** Clean up leftover test scripts and organize them into a dedicated directory.

**Implementation:**
Moved `json_to_srt.py`, `combine_subtitles.py`, and various `compare_audio-test.*` artifacts to the `test/` directory. This keeps the root directory clean and separates utility/test scripts from the main package code.

