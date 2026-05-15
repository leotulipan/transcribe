# Recent-trends audit: problems to fix

## Context
Review of the last ~40 days of work (commits `94dbbff` 2026-02-04 → `f6d0722` 2026-03-19, plus uncommitted edits) to surface latent bugs. User reports a concrete symptom: *"on some API calls it says the size is not correct, it says it resizes but then it doesn't resize and uploads the original."* This audit confirms that symptom has a root cause in `optimize_audio_for_api`, plus a few smaller issues worth fixing in the same pass.

## Findings

### 1. Silent passthrough when compression fails (matches user's bug) — HIGH
**File:** `audio_transcribe/transcribe_helpers/audio_processing.py:560-792`

The cascade `passthrough → extract → FLAC → MP3` has three silent-fallthrough paths that result in the **original (oversize) file being returned** while logs claim resize was attempted:

- **FLAC path (L691-733):** when `flac_success == False` (both PyAV and ffmpeg fallback fail), the outer `if flac_success and flac_path.exists():` block is skipped. `current_path` stays on the *original* input. Only a `debug`-level line is emitted ("PyAV FLAC conversion failed, using ffmpeg fallback") — no warning surfaces.
- **MP3 path (L735-780):** same pattern — both backends can fail silently, the `if mp3_success and mp3_path.exists():` block is skipped, no warning.
- **Final fallback (L782-792):** returns `OptimizationResult(path=current_path, is_temporary=is_current_temp, ...)` with *no guard* that the returned file actually fits the API limit. For Groq/OpenAI the caller (`cli.py:271`) then invokes chunking via `ChunkingMixin`, which rescues the flow but masks the underlying conversion failure; for APIs without chunking it errors out late. The misleading "resizes then uploads original" feedback is this path.

**Fix direction:**
- Promote the "PyAV X failed, trying ffmpeg" messages from `debug` to `warning` when *both* backends fail.
- In the final-fallback return (L782), if `final_size_mb > max_size_mb`, log an explicit `logger.error` listing which strategies were attempted and why they failed, so the caller's chunking rescue doesn't hide the real failure.
- Track a `List[str]` of attempted strategies inside `optimize_audio_for_api` and include it in the returned `OptimizationResult` (new field) for caller-side diagnostics.

### 2. Same ffmpeg-availability gap in FLAC/MP3 legacy fallbacks — MEDIUM
The PyAV branches check `is_pyav_available()`; the ffmpeg fallback branches call `convert_to_flac()` / `convert_to_mp3()` directly without calling `require_ffmpeg()` first (only extraction does, L641). On systems without ffmpeg the fallback silently returns `None` and contributes to issue #1. Add `require_ffmpeg()` before invoking the legacy helpers, or have those helpers raise a clear error.

### 3. `desktop.ini` polluting `.git/refs/` — LOW
Every git command prints `warning: bad replace ref name: desktop.ini`. Google Drive drops `desktop.ini` into `.git/refs/replace/`. One-shot cleanup: `git for-each-ref --format='%(refname)' refs/replace | grep desktop && rm .git/refs/replace/desktop.ini`; long-term mitigation is a `.gitattributes`/hook note 
also at desktop.ini to .gitignore

### 4. Uncommitted scratch files on main — LOW
commit but rename/sort first
Working tree has `research.md`, `docs/assembly_ai_filler_words_prompt_ideas.txt`, `.claude/plans/` untracked, plus modified `CLAUDE.md`, `settings.local.json`, `Transcribe Elevenlabs (YouTube Format).bat` and a new `Transcribe as Text.bat` / `transcribe-run.cmd`. 
gitignore:
`temp/`
remove:
`temp_help.txt`

### 5. Gemini SDK deprecation (known, still open) — MEDIUM
Memory notes `google.generativeai` is deprecated in favor of `google.genai`. Not regressed recently but worth scheduling before Google removes the old package.

## Recommended immediate fix (issue #1)

**File to modify:** `audio_transcribe/transcribe_helpers/audio_processing.py`

1. Add `attempted_strategies: List[str]` and `failure_reason: Optional[str]` to `OptimizationResult` (L407-).
2. In each conversion branch, append the strategy name and log `warning` on both-backends-failed.
3. At the final-fallback return (L782), if size still exceeds `max_size_mb`, emit `logger.error(f"Optimization could not bring file under {max_size_mb}MB after trying {attempted_strategies}. Uploading original — expect API-side rejection.")`.
4. Add a unit test in `tests/unit/` that monkeypatches `convert_to_flac_pyav`, `convert_to_flac`, `convert_to_mp3_pyav`, `convert_to_mp3` to all return `None`/`False`, and asserts:
   - `OptimizationResult.size_mb > max_size_mb` is surfaced,
   - `attempted_strategies` contains `["flac", "mp3"]`,
   - an error-level log record was emitted.

## Verification
- `uv run python -m pytest tests/unit/ -v` — new test passes; existing suite green.
- Manual: point CLI at a ≥30 MB audio file using `--api groq` with PyAV disabled and ffmpeg removed from PATH; confirm the log now shows a visible error instead of silently uploading the original.
