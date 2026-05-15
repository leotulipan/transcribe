# Plan: Gemini SDK Migration (`google-generativeai` → `google-genai`)

## Context
`google-generativeai` is deprecated (support ends ~Sept 2025) in favor of the unified `google-genai` SDK. The project currently uses the old SDK in `audio_transcribe/utils/api/gemini.py`. Migrating before the deprecation hits keeps Gemini transcription working, unblocks newer models (e.g. `gemini-2.5-*`, `gemini-3-*`), and removes an EOL dependency.

## Scope
Single file migration plus dependency/build metadata. No behavioral change to the CLI, output formats, or `TranscriptionResult` contract.

## Key API differences

| Concern | Old (`google.generativeai`) | New (`google.genai`) |
|---|---|---|
| Import | `import google.generativeai as genai` | `from google import genai` / `from google.genai import types` |
| Init | `genai.configure(api_key=...)` | `client = genai.Client(api_key=...)` |
| List models | `genai.list_models()` → has `.supported_generation_methods` | `client.models.list()` → check `supported_actions` contains `"generateContent"` |
| Generate | `genai.GenerativeModel(model).generate_content(content)` | `client.models.generate_content(model=model, contents=[...])` |
| Inline audio | dict `{"inline_data": {"mime_type", "data"(b64)}}` | `types.Part.from_bytes(data=raw_bytes, mime_type=...)` (no base64 needed) |
| File upload | `genai.upload_file(path=, display_name=, mime_type=)` → `.uri` | `client.files.upload(file=path, config=types.UploadFileConfig(mime_type=..., display_name=...))` → pass file object directly to `contents=[my_file, prompt]` |

## Files to modify

### 1. `audio_transcribe/utils/api/gemini.py` (primary)
- Replace import block (lines 49–60) with `from google import genai; from google.genai import types` and `self.client = genai.Client(api_key=self.api_key)`. Drop `self.genai`.
- `list_models()` (62–78): iterate `self.client.models.list()`; filter by `"generateContent" in (m.supported_actions or [])` and `"flash" in m.name.lower()`.
- `check_api_key()` (80–99): unchanged logic, just relies on new `list_models()` / `self.client`.
- `_transcribe_inline()` (138–209): drop base64; use `types.Part.from_bytes(data=f.read(), mime_type=mime_type)`. Call `self.client.models.generate_content(model=model, contents=[part, prompt])`. `response.text` is the same.
- `_transcribe_with_files_api()` (211–281): upload via `self.client.files.upload(...)`, pass the returned file object directly in `contents=[uploaded_file, prompt]` (no need to extract `file_uri` for a dict). Store `uploaded_file.name`/`uploaded_file.uri` in `raw_data` for the JSON.
- `_upload_file()` (283–313): replace with `client.files.upload(file=audio_path, config=types.UploadFileConfig(display_name=..., mime_type=...))`. Return the file object (or adjust signature to return `(file_obj, uri)`).
- Keep `_get_mime_type`, `_get_language_name`, `AUDIO_MIME_TYPES`, retry wrapper (`self.with_retry`), `save_result`, prompt strings, and `TranscriptionResult` construction unchanged.

### 2. `pyproject.toml`
- Line 24: `"google-generativeai>=0.8.0"` → `"google-genai>=1.0.0"`.
- Line 55 (PyInstaller hidden imports): `"google.generativeai"` → `"google.genai"`.

### 3. `build.py`
Grep showed no matches, but double-check for any `generativeai` string before building.

### 4. Memory
Remove or update the "Gemini deprecated SDK" line in `MEMORY.md` after the migration lands.

## Out of scope
- `build/` and `build/bdist.win-amd64/wheel/` copies of `gemini.py` — PyInstaller build artifacts, regenerated on next build. Do not edit.
- Adding new Gemini features (timestamps, thinking config, `gemini-3-*`). Migration only.
- New tests — no existing `tests/**/test_gemini*.py`; follow the project's existing pattern of manual verification for Gemini.

## Verification
1. `uv sync` — resolves `google-genai`, removes old package.
2. Static: `uv run python -c "from audio_transcribe.utils.api.gemini import GeminiAPI; GeminiAPI()"` — no ImportError.
3. `uv run python -m pytest tests/unit/ -v` — unit suite still green (Gemini not covered there, but ensures no import regressions).
4. Small file (≤20MB) end-to-end:
   `uv run transcribe.py test/files/sample.wav --api gemini --output text`
   → verify `.txt` produced, `_gemini.json` saved with `method: inline`.
5. Large file (>20MB) end-to-end with a longer sample → verify Files API path, upload log line, transcript output.
6. `uv run transcribe.py --api gemini --list-models` (or equivalent) — confirm `list_models()` works against new SDK.
7. `python build.py` — PyInstaller bundle succeeds, resulting exe runs `--help`.

## Risks
- **File object vs URI in `contents`**: new SDK accepts the uploaded-file object directly; passing a raw dict with `file_uri` still works but is legacy-style. Prefer the object form for forward compatibility.
- **`supported_actions` field name** differs from old `supported_generation_methods`. If the attribute is missing on older server responses, fall back to returning the hardcoded `["gemini-2.5-flash", "gemini-1.5-flash"]` list (existing behavior on exception).
- **PyInstaller hidden imports**: `google.genai` may pull submodules PyInstaller misses. If the built exe fails with ImportError at runtime, add `google.genai.types` (and any flagged submodule) to the hidden-imports list in `pyproject.toml`.
