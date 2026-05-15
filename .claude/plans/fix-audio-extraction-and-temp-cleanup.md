# Fix: Audio extraction failure + temp dir cleanup

## Context
A 248MB MP4 video fails to transcribe with AssemblyAI (200MB limit). Audio extraction silently fails because the output path uses the video extension (`.mp4`), so subsequent FLAC/MP3 conversions start from the full video file and produce oversized outputs. Additionally, temp directories are left behind on error paths.

## Root Causes

### 1. Wrong extension for extracted audio (`audio_processing.py:623`)
```python
extracted_path = manager.get_path_for(FileOperation.EXTRACTED, current_path.suffix[1:])
```
Uses `.mp4` as the output extension. PyAV tries to write audio-only data into an MP4 container → fails. Fix: detect the audio codec and use an appropriate audio extension (e.g., `.aac`, `.m4a`, `.ogg`).

### 2. No cleanup on early return (`cli.py:284-287`)
When optimization returns a file exceeding the limit and API doesn't support chunking, `return []` skips cleanup. The `opt_result.intermediate_manager` is never cleaned up, and the temp dir persists.

### 3. AssemblyAI lacks ChunkingMixin (separate consideration)
AssemblyAI doesn't support chunking, but with bug #1 fixed, extraction should produce audio well under 200MB, so this isn't the immediate issue. Could be added later for very large files.

## Changes

### File 1: `audio_transcribe/transcribe_helpers/audio_processing.py`
**Line 623** — Change extracted audio extension from video extension to proper audio extension:
- Inspect the input container's audio codec via PyAV (or assume `.m4a` for MP4 video, `.ogg` for webm, etc.)
- Simplest fix: map video extensions to audio container extensions:
  - `.mp4`, `.mov`, `.m4v` → `m4a`
  - `.mkv`, `.webm` → `ogg`
  - `.avi` → `m4a` (safe default)
- Also fix ffmpeg fallback (line 640-656): it already uses `.m4a` internally but moves to the wrong-extension path

### File 2: `audio_transcribe/cli.py`
**Lines 284-287** — Add cleanup before early return:
```python
elif not opt_result.fits_limit(max_size_mb):
    if opt_result.intermediate_manager:
        opt_result.intermediate_manager.cleanup()
    logger.error(...)
    return []
```

Also add cleanup in the `finally` block for the intermediate manager (in case of any unhandled path):
```python
finally:
    if 'opt_result' in locals() and opt_result and opt_result.intermediate_manager:
        opt_result.cleanup()
```

## Verification
1. Run the same command with `-v` and verify:
   - Strategy 1 extracts to `.m4a` (not `.mp4`)
   - Extraction succeeds and file is small enough, OR FLAC/MP3 conversion works from the extracted audio
   - No temp dirs left behind after success or failure
2. Test with a small video file to ensure the happy path still works
