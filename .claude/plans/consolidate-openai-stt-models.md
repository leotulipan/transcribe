# Plan: Consolidate OpenAI STT Models & Streamline Infrastructure

## Context

The project has **two separate OpenAI API classes** (`OpenAIAPI` in `openai.py` and `OpenAIExtendedAPI` in `openai_extended.py`) which is confusing and leaves the GPT-4o text models untested/broken. The user's API key now works with all OpenAI STT models. Goals:

1. Merge into one `OpenAIAPI` class handling all 4 models
2. Make all models work and test them
3. Streamline the infrastructure (move per-model capabilities into `MODEL_REGISTRY`, add `get_model_capabilities()` helper) so adding new APIs/models is easier

**OpenAI STT models (from official docs):**
| Model | Response Formats | Word Timestamps | Diarization | Needs FLAC |
|---|---|---|---|---|
| `whisper-1` | verbose_json, json, text, srt, vtt | Yes (word+segment) | No | Yes |
| `gpt-4o-transcribe` | json, text | No | No | No |
| `gpt-4o-mini-transcribe` | json, text | No (best WER) | No | No |
| `gpt-4o-transcribe-diarize` | json, text, diarized_json | Yes (via diarized_json) | Yes | No |

All models: 25MB file size limit, accept mp3/wav/m4a/flac/webm/ogg.

---

## Part A: OpenAI Merge (Steps 1-4)

### Step 1: Rewrite `audio_transcribe/utils/api/openai.py`

Merge all logic from both files into a single `OpenAIAPI` class.

**Key changes:**
- Keep `ChunkingMixin` inheritance
- Add `MODEL_CAPABILITIES` dict with per-model metadata:
  - `requires_flac`: `True` only for `whisper-1`
  - `default_response_format`: `verbose_json` / `json` / `diarized_json`
  - `supports_word_timestamps`, `supports_segment_timestamps`, `supports_speaker_diarization`
  - `timestamp_granularities`: `["word"]` for whisper-1, `[]` for others
- **Fix `transcribe_chunk` signature** to match `ChunkingMixin`: `(self, chunk_path: Path, chunk_index: int, start_time: float, **kwargs)` returning `Tuple[Dict, float]` (current signature is wrong — uses `chunk_start_ms` in ms)
- **Fix `merge_chunk_results`** to return `Dict` (not `TranscriptionResult`) — follow Groq pattern
- Make FLAC conversion **model-conditional** inside `transcribe()`: only for `whisper-1`
- Select `response_format` based on model capabilities (verbose_json / json / diarized_json)
- Keep `_parse_response`, `_parse_diarization_response`, `_parse_segment_response` from `openai_extended.py`
- Handle `diarized_json` response which may return `utterances` list (not flat `words`)
- Default model: `gpt-4o-mini-transcribe`
- `list_models()`: return static `MODEL_CAPABILITIES.keys()` (no live API call needed)
- Add `get_model_capabilities(model)` public method for CLI to query

**Reuse from existing code:**
- `openai.py`: `__init__`, `check_api_key`, FLAC conversion logic, chunking flow, JSON save logic
- `openai_extended.py`: `MODEL_CAPABILITIES`, `_parse_response`, `_parse_diarization_response`, `_parse_segment_response`, capability-based format selection

### Step 2: Delete `audio_transcribe/utils/api/openai_extended.py`

### Step 3: Update `audio_transcribe/utils/api/__init__.py`

- Remove `OpenAIExtendedAPI` import, factory branch, and `__all__` entry

### Step 4: Update `audio_transcribe/transcribe_helpers/audio_processing.py`

- Change `API_FORMAT_REQUIREMENTS["openai"]["requires_flac"]` from `True` to `False`
- FLAC conversion is now model-driven inside `OpenAIAPI.transcribe()` (only for `whisper-1`)

---

## Part B: Infrastructure Streamlining (Steps 5-8)

### Step 5: Enhance `audio_transcribe/utils/models.py` — Central Capability Registry

Move per-model capabilities into `MODEL_REGISTRY` so any code can query model features without importing API classes. Add a `get_model_capabilities()` helper.

```python
MODEL_REGISTRY = {
    "openai": {
        "default": "gpt-4o-mini-transcribe",
        "models": {
            "gpt-4o-mini-transcribe": {
                "supports_word_timestamps": False,
                "supports_speaker_diarization": False,
                "requires_flac": False,
                "note": "Best WER, text-only"
            },
            "gpt-4o-transcribe": {
                "supports_word_timestamps": False,
                "supports_speaker_diarization": False,
                "requires_flac": False,
            },
            "gpt-4o-transcribe-diarize": {
                "supports_word_timestamps": True,
                "supports_speaker_diarization": True,
                "requires_flac": False,
            },
            "whisper-1": {
                "supports_word_timestamps": True,
                "supports_speaker_diarization": False,
                "requires_flac": True,
                "note": "Word+segment timestamps, legacy"
            },
        },
        "env_key": "OPENAI_API_KEY",
    },
    "groq": {
        "default": "whisper-large-v3",
        "models": {
            "whisper-large-v3": {"supports_word_timestamps": True, "supports_speaker_diarization": False},
            "whisper-large-v3-turbo": {"supports_word_timestamps": True, "supports_speaker_diarization": False},
            "distil-whisper-large-v3-en": {"supports_word_timestamps": True, "supports_speaker_diarization": False},
        },
        "env_key": "GROQ_API_KEY",
    },
    # ... same pattern for assemblyai, elevenlabs, gemini, mistral
}

def get_model_capabilities(api_name: str, model: str) -> dict:
    """Get capabilities for a specific model. Returns empty dict if not found."""
    api = MODEL_REGISTRY.get(api_name, {})
    models = api.get("models", {})
    if isinstance(models, dict):
        return models.get(model, {})
    return {}

def get_available_models(api_name: str) -> list:
    """Get list of available model names for a given API."""
    models = MODEL_REGISTRY.get(api_name, {}).get("models", {})
    if isinstance(models, dict):
        return list(models.keys())
    return list(models)  # backward compat if still a list
```

This is a **structural change** to `models` from `list` to `dict`. All callers of `get_available_models()` already expect a list and the function still returns one. The `get_model_capabilities()` function is new.

### Step 6: Update `audio_transcribe/utils/defaults.py`

- Change `openai` default model from `whisper-1` to `gpt-4o-mini-transcribe`
- Add missing API defaults for `gemini` and `mistral`

```python
API_DEFAULTS = {
    'groq': {'model': 'whisper-large-v3'},
    'openai': {'model': 'gpt-4o-mini-transcribe'},
    'assemblyai': {'model': 'universal-3-pro', 'speaker_labels': True, 'disfluencies': True},
    'elevenlabs': {'model_id': 'scribe_v1'},
    'gemini': {'model': 'gemini-2.0-flash-exp'},
    'mistral': {'model': 'voxtral-mini-2507'},
}
```

### Step 7: Update `audio_transcribe/cli.py`

- Update `--api` help text: `"API to use (groq, openai, assemblyai, elevenlabs, gemini, mistral)"`
- Make the `supports_word_timestamps` warning (line 238) model-aware:
  ```python
  # Use the central registry instead of class-level flag
  from audio_transcribe.utils.models import get_model_capabilities
  effective_model = kwargs.get('model') or get_default_model(api_name)
  caps = get_model_capabilities(api_name, effective_model)
  if caps and caps.get('supports_word_timestamps') is False and any(...):
      logger.warning(...)
  ```

### Step 8: Update integration tests

Update `tests/integration/test_output_format_fallback.py` and `tests/integration/test_model_capabilities.py`:
- Change all `"openai_extended"` references to `"openai"`
- Import `MODEL_CAPABILITIES` from `audio_transcribe.utils.api.openai` instead of `openai_extended`
- Remove `OpenAIExtendedAPI` imports

---

## Files Modified (in order)

1. `audio_transcribe/utils/api/openai.py` — rewrite (merge both classes)
2. `audio_transcribe/utils/api/openai_extended.py` — delete
3. `audio_transcribe/utils/api/__init__.py` — remove openai_extended
4. `audio_transcribe/transcribe_helpers/audio_processing.py` — openai requires_flac→False
5. `audio_transcribe/utils/models.py` — unified registry with per-model capabilities
6. `audio_transcribe/utils/defaults.py` — update defaults
7. `audio_transcribe/cli.py` — help text + model-aware warning
8. `tests/integration/test_output_format_fallback.py` — update refs
9. `tests/integration/test_model_capabilities.py` — update refs

---

## Verification

1. `uv run transcribe.py --list` — confirm all 4 OpenAI models appear under single "openai" entry
2. Test each model with a short audio file:
   ```bash
   uv run transcribe.py test/files/sample.wav --api openai --model whisper-1 -o text,srt
   uv run transcribe.py test/files/sample.wav --api openai --model gpt-4o-mini-transcribe -o text
   uv run transcribe.py test/files/sample.wav --api openai --model gpt-4o-transcribe -o text
   uv run transcribe.py test/files/sample.wav --api openai --model gpt-4o-transcribe-diarize -o text
   ```
3. Verify `--api openai` without `--model` uses `gpt-4o-mini-transcribe` by default
4. Verify `--api openai_extended` gives clean `ValueError: Unknown API`
5. Verify SRT output with `whisper-1` still has proper word timestamps
6. Verify text-only models produce approximate SRT via `generate_words_from_text`
7. Verify `get_model_capabilities()` works from CLI for all APIs
