# Plan: Two-level force flag + skip feedback

## Context
When processing folders, files that already have transcription outputs (JSON, SRT, TXT) are either silently skipped or re-processed without clear user feedback. The user wants:
1. Clear skip messages even without `--verbose` (especially in folder mode)
2. Skip files when SRT/TXT already exists (not just JSON)
3. Two levels of force: regenerate outputs from existing JSON vs full re-transcription

## Current State
- `--force` (`-r`): disables JSON reuse check, always calls API
- API-specific JSON reuse: if `{name}_{api}.json` exists and no `--force`, skips API call and regenerates outputs
- `check_transcript_exists()` is defined but **never called** — no SRT/TXT skip logic
- Skip messages use `logger.info` (only visible with `--verbose`)

## Design

### Two-level force: `--regenerate` and `--force`

| Flag | Behavior |
|------|----------|
| *(neither)* | Skip if `.srt` exists (only SRT triggers skip). If only JSON exists, regenerate outputs from JSON. If nothing exists, call API. |
| `--regenerate` | Ignore existing `.srt`. Regenerate outputs from existing JSON if available, otherwise call API. |
| `--force` (`-r`) | Ignore everything. Always call API and regenerate all outputs. |

### Skip feedback (always visible)

Use `logger.warning()` or a dedicated print for skip messages so they show at default INFO level:
- `"[SKIP] filename.mp4 — SRT already exists (use --regenerate to recreate from JSON, --force to re-transcribe)"`
- `"[REUSE] filename.mp4 — regenerating outputs from existing JSON"`

### Files to modify

1. **`audio_transcribe/cli.py`**
   - Add `--regenerate` flag (keep `--force`/`-r` as-is)
   - Activate SRT existence check (adapt `check_transcript_exists()` to only check `.srt`)
   - New logic order in `process_file()`:
     1. If `--force`: skip all checks, call API
     2. If `--regenerate`: skip SRT check, but reuse JSON if available (else call API)
     3. Default: check for existing `.srt` first → skip entirely if found; then check for JSON → reuse if found; else call API
   - Use `logger.info` with `[SKIP]`/`[REUSE]` prefixes for visibility at default log level
   - Return a status string from `process_file()`: `"processed"`, `"skipped"`, `"failed"`

2. **`audio_transcribe/cli.py` — `process_audio_path()`** (folder processing)
   - Track skip count alongside success/failed counts
   - Summary: `"5 processed, 2 skipped, 1 failed"`

## Verification
- Run on a folder with mixed state (some with JSON, some with SRT, some fresh)
- Verify default: files with existing `.srt` are skipped with visible message
- Verify `--regenerate`: SRT files are recreated from JSON, API not called
- Verify `--force`: API called for all files regardless
- Verify folder summary shows correct counts
