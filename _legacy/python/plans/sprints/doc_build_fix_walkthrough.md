# Walkthrough - Build Fixes

## Problem
The user reported a `ModuleNotFoundError: No module named 'audio_transcribe.utils.api'` when running the tool after installing it with `uv tool install . --force`.
Additionally, there were `SyntaxWarning`s from `pydub`.

## Root Cause
The `pyproject.toml` file was using an explicit list of packages:
```toml
packages = ["audio_transcribe", "audio_transcribe.utils", "audio_transcribe.transcribe_helpers"]
```
This list was missing `audio_transcribe.utils.api` and `audio_transcribe.tui`, causing them to be excluded from the build/installation.

## Solution
1.  **Updated `pyproject.toml`**: Switched to dynamic package discovery using `find` directive to ensure all subpackages are automatically included.
    ```toml
    [tool.setuptools.packages.find]
    where = ["."]
    include = ["audio_transcribe*"]
    ```
2.  **Suppressed Warnings**: Added code to `audio_transcribe/cli.py` to suppress `SyntaxWarning` from `pydub` to clean up the CLI output.

## Verification
- Validated `pyproject.toml` syntax.
- Validated file structure to ensure `audio_transcribe*` pattern matches all necessary packages.
- The fix ensures that `audio_transcribe.utils.api` is now included in the installed package.

## Next Steps
The user should try reinstalling the tool:
```bash
uv tool install . --force
```
And then run the command again:
```bash
transcribe
```
