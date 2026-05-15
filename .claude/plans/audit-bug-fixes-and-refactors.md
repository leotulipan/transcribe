# Enhancements, Refactorings & Bug Fixes

## Context
Comprehensive codebase audit found bugs, DRY violations, resilience gaps, and usability issues. This plan addresses the highest-value fixes grouped into independently committable chunks, ordered by priority.

---

## Chunk 1: Fix `transcribe_kwargs` NameError (BUG, Critical)
**File**: `audio_transcribe/cli.py`

Line 286 writes `transcribe_kwargs['chunk_length']` but `transcribe_kwargs` is only created at line 304. Large files triggering chunking will crash.

**Fix**: Move `transcribe_kwargs = kwargs.copy()` and the `original_path` assignment (lines 304-305) to before the optimization block (~line 255). Remove the duplicate at 304-305.

**Verify**: `uv run python -m pytest tests/ -v -x`

---

## Chunk 2: Fix bare `except:` clauses (BUG, Critical)
**Files**: `audio_transcribe/transcribe_helpers/audio_processing.py`, any other files with bare `except:`

**Fix**: Change `except:` → `except Exception:` everywhere. Bare except catches `SystemExit` and `KeyboardInterrupt`, breaking Ctrl+C.

**Verify**: `grep -rn "except:" audio_transcribe/ | grep -v "except Exception" | grep -v "except "` should find nothing problematic.

---

## Chunk 3: Add HTTP request timeouts (BUG, Medium)
**Files**: All `requests.get/post` calls across the codebase

**Fix**: Add `timeout=300` to every `requests.get()` and `requests.post()` call. Without this, requests can hang forever if the API is unresponsive.

**Verify**: Grep for `requests.` calls and confirm all have timeout parameter.

---

## Chunk 4: Deduplicate `list_models()` into base class (DRY)
**Files**: `audio_transcribe/utils/api/base.py` + all 6 API subclass files

**Fix**: Update base class `list_models()` to:
```python
def list_models(self) -> List[str]:
    from audio_transcribe.utils.models import get_available_models
    return get_available_models(self.api_name)
```
Remove identical overrides from subclasses that just delegate to `get_available_models()`.

**Verify**: `uv run python -m pytest tests/unit/ -v`

---

## Chunk 5: Extract `response_to_dict()` utility (DRY)
**Files**: `audio_transcribe/utils/api/base.py`, then update `groq.py`, `openai.py`, `openai_extended.py`, `mistral_voxtral.py`

**Fix**: Add static method to base class:
```python
@staticmethod
def response_to_dict(response) -> dict:
    if hasattr(response, 'model_dump'):
        return response.model_dump()
    elif hasattr(response, 'dict'):
        return response.dict()
    elif hasattr(response, '__dict__'):
        return response.__dict__.copy()
    return {"text": str(response)}
```
Replace 3-4 duplicate blocks in subclasses.

**Verify**: `uv run python -m pytest tests/ -v -x`

---

## Chunk 6: Remove duplicate `CustomJSONEncoder` (DRY)
**File**: `audio_transcribe/utils/parsers.py`

**Fix**: The `save()` method redefines `CustomJSONEncoder` that already exists at module level. Remove the inner definition and use the module-level one.

**Verify**: `uv run python -m pytest tests/unit/ -v`

---

## Chunk 7: Extract shared MIME type mapping (DRY)
**Files**: Create constant in `audio_transcribe/utils/api/base.py` or a shared location; update `gemini.py` and `mistral_voxtral.py`

**Fix**: Define `AUDIO_MIME_TYPES` dict once, import in both files.

**Verify**: `uv run python -m pytest tests/ -v -x`

---

## Chunk 8: Extract temp file cleanup helper (DRY)
**Files**: `audio_transcribe/utils/api/base.py` + all API files with duplicated cleanup

**Fix**: Add to base class:
```python
@staticmethod
def cleanup_temp_file(path):
    if path and os.path.exists(path):
        try:
            os.unlink(path)
            logger.info(f"Deleted temporary file: {path}")
        except Exception as e:
            logger.warning(f"Failed to delete temporary file {path}: {e}")
```
Replace ~4 duplicate blocks.

---

## Chunk 9: Wrap Groq/OpenAI transcription in `with_retry()` (Resilience)
**Files**: `audio_transcribe/utils/api/groq.py`, `audio_transcribe/utils/api/openai.py`

**Fix**: ElevenLabs already wraps its API call in `self.with_retry()`. Do the same for Groq and OpenAI main transcription calls for transient error resilience.

**Verify**: Integration tests with valid API keys.

---

## Chunk 10: Improve error messages with `--setup` hint (Usability)
**Files**: API `check_api_key()` methods, `cli.py` error handling

**Fix**: When API key validation fails, append: `"Run 'transcribe --setup' to configure API keys."`

---

## Verification
After all chunks:
1. `uv run python -m pytest tests/ -v` — all tests pass
2. `uv run transcribe.py --help` — CLI still works
3. Manual test: `uv run transcribe.py test/files/sample.wav --api groq` — end-to-end transcription
4. Grep for remaining bare `except:` and `requests.` without timeout

## Files Modified (Summary)
- `audio_transcribe/cli.py` — chunks 1, 10
- `audio_transcribe/utils/api/base.py` — chunks 4, 5, 7, 8
- `audio_transcribe/utils/api/groq.py` — chunks 2, 3, 5, 9
- `audio_transcribe/utils/api/openai.py` — chunks 3, 5, 9
- `audio_transcribe/utils/api/elevenlabs.py` — chunk 3
- `audio_transcribe/utils/api/gemini.py` — chunk 7
- `audio_transcribe/utils/api/mistral_voxtral.py` — chunks 5, 7
- `audio_transcribe/utils/api/assemblyai.py` — chunk 4
- `audio_transcribe/utils/parsers.py` — chunk 6
- `audio_transcribe/transcribe_helpers/audio_processing.py` — chunk 2
