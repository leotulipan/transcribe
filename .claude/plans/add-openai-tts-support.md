# Plan: Add OpenAI TTS (Text-to-Speech) Support

## Context
The project is a transcription tool (speech-to-text). The user wants to add TTS as a new capability using OpenAI's TTS models (`tts-1`, `tts-1-hd`, `gpt-4o-mini-tts`). The existing OpenAI integration only handles transcription via Whisper. The OpenAI Python client is already a dependency.

## OpenAI TTS API Reference
- **Endpoint**: `client.audio.speech.create()`
- **Models**: `tts-1`, `tts-1-hd`, `gpt-4o-mini-tts`
- **Voices**: `alloy`, `ash`, `ballad`, `coral`, `echo`, `fable`, `onyx`, `nova`, `sage`, `shimmer`, `verse`, `marin`, `cedar`
- **Formats**: `mp3`, `opus`, `aac`, `flac`, `wav`, `pcm`
- **Speed**: `0.25` to `4.0` (default `1.0`)
- **Instructions**: Text prompt to control tone (only `gpt-4o-mini-tts`)
- **Max input**: 4096 characters

## Implementation Plan

### 1. Create TTS API module
**File**: `audio_transcribe/utils/api/tts_openai.py`

New class `OpenAITTSAPI` that reuses the existing OpenAI client pattern from `openai.py`:
- `__init__`: Initialize OpenAI client (same pattern as `OpenAIAPI`)
- `speak(text, output_path, model, voice, format, speed, instructions)`: Generate speech, stream to file
- `check_api_key()`: Reuse same validation
- For texts > 4096 chars: split into chunks and concatenate audio files using pydub

### 2. Add TTS CLI subcommand
**File**: `audio_transcribe/cli.py`

Add a `tts` command under the existing `tools` group:
```
transcribe tools tts "Hello world" --voice coral --model gpt-4o-mini-tts --format mp3 --speed 1.0 --instructions "Speak cheerfully" -o output.mp3
```

Options:
- `text` (argument) — text to speak, OR `--file` to read from a text file
- `--voice` / `-V` — voice name (default: `alloy`)
- `--model` / `-m` — TTS model (default: `tts-1`)
- `--format` / `-f` — output format (default: `mp3`)
- `--speed` / `-s` — speed 0.25-4.0 (default: `1.0`)
- `--instructions` / `-i` — voice instructions (only gpt-4o-mini-tts)
- `--output` / `-o` — output file path (default: `speech.<format>`)
- `--list-voices` — list available voices

### 3. Add TTS models to registry
**File**: `audio_transcribe/utils/models.py`

Add `openai_tts` entry:
```python
"openai_tts": {
    "default": "tts-1",
    "models": ["tts-1", "tts-1-hd", "gpt-4o-mini-tts"],
    "voices": ["alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer", "verse", "marin", "cedar"],
    "note": "gpt-4o-mini-tts supports voice instructions"
}
```

### 4. Update `--list` output
**File**: `audio_transcribe/cli.py`

Include TTS models in the `--list` output so users can see available TTS models and voices.

## Files to Modify/Create
- **Create**: `audio_transcribe/utils/api/tts_openai.py`
- **Modify**: `audio_transcribe/cli.py` — add `tts` subcommand under `tools`
- **Modify**: `audio_transcribe/utils/models.py` — add TTS models/voices

## Verification
1. `uv run transcribe.py tools tts "Hello, this is a test." --voice coral --model tts-1 -o temp/test_tts1.mp3`
2. `uv run transcribe.py tools tts "Hello, this is a test." --voice alloy --model tts-1-hd -o temp/test_tts1hd.mp3`
3. `uv run transcribe.py tools tts "Hello, this is a test." --voice nova --model gpt-4o-mini-tts --instructions "Speak in a warm, friendly tone" -o temp/test_mini.mp3`
4. Test with `--file` flag reading a .txt file
5. Test `--list-voices` flag
6. Verify all 3 output files are valid audio and play correctly
