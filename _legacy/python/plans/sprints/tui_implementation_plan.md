# TUI Implementation Plan

## Goal
Create a robust Text User Interface (TUI) for the Audio Transcribe tool to handle:
1.  **Setup & Configuration**: Interactive setup for API keys and endpoints.
2.  **Model Discovery**: Display available models for each provider (hardcoded fallback if no discovery endpoint).
3.  **Interactive Execution**: When run without arguments (e.g., drag-and-drop), show a TUI to confirm/edit settings before processing.
4.  **Defaults Management**: Allow users to set and persist default preferences (language, output format, etc.).

## Technologies
- **Library**: `questionary` (for interactive prompts) or `rich` (for beautiful output) + `click` (integration). Given the requirement for a "really solid" TUI, `textual` might be overkill for a configuration wizard but `questionary` is excellent for linear wizards. `rich` is essential for nice formatting.
- **Configuration**: Store settings in a user-local config file (e.g., `~/.audio_transcribe/config.json` or `.env` in home dir).

## Proposed Features

### 1. Setup Wizard (`transcribe setup`)
- **API Key Entry**: Interactive prompts to enter/update API keys for AssemblyAI, ElevenLabs, Groq, OpenAI.
- **Validation**: Immediate validation of keys using the `check_api_key` methods.
- **Storage**: Save keys to a secure location (or `.env` file).

### 2. Interactive Mode (No Args / Drag-and-Drop)
- **Detection**: If `transcribe` is called without arguments but with a file path (via drag-and-drop, often passed as arg 1), or just no args.
- **Flow**:
    1.  **File Detection**: Show detected file(s).
    2.  **API Selection**: "Which service do you want to use?" (Radio button list).
    3.  **Model Selection**: Dynamic list based on chosen API.
    4.  **Language**: "Auto" or select from common languages.
    5.  **Output Format**: Checkbox list for formats (Text, SRT, DaVinci, etc.).
    6.  **Confirmation**: Summary of settings -> [Run] / [Edit].

### 3. Model Management
- **Hardcoded Lists**: Maintain a `models.py` with known models for each provider (e.g., Groq: `whisper-large-v3`, AssemblyAI: `best`, `nano`).
- **Dynamic Fetch**: If API supports it (e.g., OpenAI `models.list`), fetch and cache.

### 4. Configuration Persistence
- **Defaults**: Store user preferences for:
    - Default API
    - Default Language
    - Default Output Formats
- **Load Order**: CLI Args > Config File > Hardcoded Defaults.

## Implementation Steps

1.  **Dependency**: Add `questionary` and `rich` to `pyproject.toml`.
2.  **Config Manager**: Create `utils/config.py` to handle loading/saving JSON config/env vars.
3.  **Model Registry**: Create `utils/models.py` with dictionaries of available models.
4.  **TUI Module**: Create `tui/` package.
    - `wizard.py`: Setup wizard logic.
    - `interactive.py`: Main interactive flow.
5.  **CLI Integration**: Update `cli.py`:
    - Add `setup` command implementation.
    - In `main`, if no args provided (or just file arg), trigger `interactive_mode`.

## User Review Required
- **Library Choice**: Is `questionary` acceptable? It's lightweight and robust for wizards. `Textual` is a full TUI framework (like a GUI in terminal) which might be too heavy but offers a "dashboard" feel. **Recommendation**: Start with `questionary` + `rich` for a polished CLI experience.
- **Config Location**: Defaulting to user home directory `~/.audio_transcribe/` to avoid cluttering the script folder.

## Proposed Changes

### [NEW] `audio_transcribe/utils/config.py`
- `ConfigManager` class.

### [NEW] `audio_transcribe/utils/models.py`
- `MODEL_REGISTRY` constant.

### [NEW] `audio_transcribe/tui/`
- `__init__.py`
- `wizard.py`
- `interactive.py`

### [MODIFY] `audio_transcribe/cli.py`
- Import TUI modules.
- Implement `setup` command.
- Add logic to trigger interactive mode in `main`.

### [MODIFY] `pyproject.toml`
- Add `questionary`, `rich`.
