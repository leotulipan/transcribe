# Plan: UX Fixes — Interactive Triggering, Wildcards, .mov, Error Reporting, SRT Checkbox

## Context

User ran `transcribe -o text .` in a folder of iPhone `.mov` exports. Several issues surfaced:

1. Interactive mode triggered even though a path + output format were passed.
2. Folder scan reported "No audio files found" despite 10+ `.mov` files present.
3. `transcribe -o text *.mov` (and quoted) failed with Click "No such command" errors.
4. Video → audio extraction silently failed with only a generic warning; no reason, no final success/failure summary for the file.
5. In the interactive output-format checkbox, `srt` appears always pre-checked and (per user) cannot be toggled off; `-o text` from the CLI was not honored.

## Root Causes (verified in code)

| # | Symptom | Root cause | Location |
|---|---|---|---|
| 1 | Interactive triggers with only `-o` | `should_run_interactive = True` whenever `--api` is absent, regardless of other flags. | `audio_transcribe/cli.py:586-592` |
| 2 | `.mov` folder returns "No audio files found" | Default extension list omits `.mov`. | `cli.py:397` (`['.mp3', '.wav', '.m4a', '.mp4', '.mkv', '.flac', '.ogg', '.webm']`) |
| 3 | `*.mov` → "No such command '…mov'" | PowerShell 7+ / wrapper expands the glob into multiple args; Click's `@click.group` treats extras beyond `input_path` as subcommand names. The CLI never accepts multiple positionals or explicit glob patterns. | `cli.py:440-441` (`@click.group(invoke_without_command=True)`, single `input_path` arg) |
| 4 | "Failed to extract audio…" with no reason, no final status | `extract_audio_from_mp4` returns `None` on any failure; only a generic `logger.warning` is emitted — no stderr from ffmpeg/PyAV, no exception surfaced. `.mov` from iPhone uses HEVC + possibly sidecar audio; extractor may reject. Also: no terminal per-file SUCCESS line when single file processed (only inside `process_audio_path` folder branch). | `elevenlabs.py:85-99`, `transcribe_helpers/audio_processing.py` (extract fn), `cli.py:387-389` |
| 5 | SRT "always checked, not unselectable" + `-o text` ignored | `run_interactive_mode` builds `format_choices` from config defaults and **ignores the CLI `-o` the user already passed**. Also, `-o` from the outer command is only applied *after* interactive returns, and interactive overwrites it. The "can't uncheck" perception is likely users not knowing `<space>` toggles questionary checkboxes — but the fact that we prompt at all when `-o` is explicit is the real bug. | `tui/interactive.py:135-154`, `cli.py:594-617` |

## Acceptance Criteria

- [ ] `transcribe -o text .` in a folder of `.mov` files: runs non-interactively using config defaults for API/model/language, finds all `.mov` files, writes only `.txt` outputs (no `.srt`).
- [ ] `.mov` is part of the default video extension set; folder scan finds `.mov`, `.mp4`, `.mkv`, `.avi`, `.webm` consistently.
- [ ] `transcribe -o text *.mov` (shell-expanded to N files) either (a) processes all matched files, or (b) fails with a clear error message explaining that multiple positional arguments aren't supported and suggesting passing the folder instead. No cryptic "No such command".
- [ ] When ElevenLabs audio extraction fails, the log line includes the underlying reason (ffmpeg/PyAV error message) at WARNING, and a terminal per-file status (SUCCESS / FAILED) is emitted for single-file processing, matching the folder summary.
- [ ] When `-o` is passed on the CLI and interactive mode does still launch (e.g. missing API only), the selection is pre-populated from the CLI, not from config defaults; outputs chosen are respected (no forced SRT).
- [ ] Interactive mode is only triggered when truly needed: no path **and** no API configured. Passing a path with any explicit flag (`-o`, `-a`, `-l`, `-m`) should NOT launch interactive unless data is still missing.

## Technical Approach

### 1. Fix interactive-mode trigger logic (`cli.py`)

Replace current "trigger if no --api" rule with: trigger interactive only if **target_path is missing**. Resolve `api` from `--api` → config `default_api` → hard default (`groq`) silently. Today lines 591-592 force interactive any time `--api` is absent, which is the main UX bug.

```python
should_run_interactive = target_path is None
# Remove: `if target_path and not api: should_run_interactive = True`
```

API resolution already has fallback at lines 620-622 — just let it run.

### 2. Add `.mov` + `.avi` to the default folder-scan extensions (`cli.py:397`)

```python
extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg',
              '.mp4', '.mkv', '.mov', '.avi', '.webm']
```

Consider centralizing in a module constant (`SUPPORTED_EXTENSIONS` in `transcribe_helpers/audio_processing.py`) since `elevenlabs.py:85` has its own hard-coded list — they drift.

### 3. Handle shell-expanded multi-argument invocations (`cli.py`)

Two options, pick one:

- **A (minimal):** Change `input_path` from `click.argument("input_path", required=False)` to `nargs=-1` and iterate over all paths. Each path runs through `process_audio_path`.
- **B (conservative):** Keep single arg, but detect when Click raises "No such command" for a path-shaped token and emit a friendly error: *"Multiple files passed via shell glob. Pass the folder instead, or quote the pattern."*

Recommendation: **A.** Trivial change, and matches user mental model. Need to handle `@click.group(invoke_without_command=True)` — switching the argument to `nargs=-1` conflicts with subcommand dispatch. Simplest resolution: demote `tools` subgroup to a separate top-level command or use a sentinel (e.g. input_path must not be `"tools"`). Since `tools join-srt` is unimplemented (cli.py:704-706), we can drop the subgroup entirely and make `main` a plain `@click.command`. Reduces complexity.

### 4. Surface extraction failure reason (`elevenlabs.py` + `audio_processing.py`)

- Change `extract_audio_from_mp4` to raise instead of returning `None` (or return `(path, error_msg)`).
- `elevenlabs.py:99` logs the actual stderr/exception text.
- Add a single-file summary line in `cli.py:process_audio_path` for `is_file()` branch mirroring the folder summary (lines 430-435): `[SUMMARY] 1 processed` / `1 failed`.

### 5. Honor CLI `-o` AND make SRT truly deselectable in the TUI checkbox (`tui/interactive.py`)

- `run_interactive_mode(file_path, cli_output=None)` — pass `output` from `cli.py:598` into the wizard.
- If `cli_output` is non-empty, **skip** the checkbox step entirely and use it verbatim.
- Otherwise, use it as the `checked=` seed instead of config defaults.
- Also pass `cli_api`, `cli_language`, `cli_model` to skip those prompts when already supplied.

**Checkbox deselection fix** (`tui/interactive.py:135-154`): the user reports that `srt` appears pinned and cannot be unchecked, so selecting only `text` is impossible via the TUI. Concrete changes:

- **Print a usage hint** above the checkbox prompt: `"(use <space> to toggle, <enter> to confirm)"`. Questionary's default footer is easy to miss on PowerShell.
- **Stop forcing srt on in the "empty selection" fallback** (line 150-152). Today an empty selection silently becomes `["text", "srt"]`, which reinforces the "can't get rid of srt" feeling. Replace with: if user confirms empty selection, re-prompt once, else treat as abort.
- **Verify the `Choice(..., checked=...)` values are actually togglable** — questionary on Windows conhost/PowerShell has had space-bar edge cases; if we hit one, fall back to a `questionary.select` with multi-select via comma-separated text input as a last resort. (Add a simple integration check: launch wizard, select only `text`, assert returned `options["output"] == ["text"]` and no `.srt` is written.)
- **Sanity-check downstream**: confirm `DefaultsManager.get_effective_params` (`cli.py:684`) does not re-inject `srt` when the user chose only `text`. Grep shows it doesn't touch `output`, but add a unit test to lock this in so a future default doesn't silently re-add srt.

Acceptance for this sub-fix: running `transcribe` with no flags, pointing at a `.mov`, selecting **only** `text` in the checkbox → only `.txt` is written, no `.srt` appears on disk.

### 6. Documentation / help

- Update CLI help: mention that multiple paths via shell glob are supported (`nargs=-1`).
- Add a short line to README: "Pass a folder or multiple files; wildcards are expanded by your shell."

## Testing Strategy

Unit tests (tests/unit):

- `test_cli_interactive_trigger.py`: assert interactive only triggers when path is missing.
- `test_cli_extension_defaults.py`: `.mov`/`.avi` discovered in a tmp folder.
- `test_cli_multi_path.py`: invoke CLI with two fake files; both reach `process_audio_path`.
- `test_interactive_respects_cli_output.py`: when `cli_output=['text']` is passed, checkbox step is skipped.

Manual verification on the reporter's folder (`…\Story Format Videos`):

1. `transcribe -o text .` → non-interactive, processes all `.mov`, writes only `.txt`.
2. `transcribe -o text *.mov` (PowerShell) → all files processed.
3. Single broken `.mov`: error log names the actual ffmpeg reason; final line says `[SUMMARY] 0 processed, 1 failed`.

## Risks & Mitigations

- **Risk:** Changing `main` from `@click.group` to `@click.command` breaks the undocumented `tools join-srt` subcommand. **Mitigation:** It's a stub (`click.echo("not implemented yet")`); remove it, or re-add as a sibling command later.
- **Risk:** `nargs=-1` with existing `NormalizedPath(exists=False)` validation — need `type=NormalizedPath(...)` applied per item. Verify Click handles this.
- **Risk:** Skipping interactive prompts when CLI flags are provided may surprise users who relied on the interactive flow even with partial flags. **Mitigation:** log `"Using --api X from CLI"` etc. so behavior is visible.

## Out of Scope

- Reworking the interactive wizard UX beyond the CLI-override fix.
- Adding shell-side glob expansion on Windows (PowerShell already expands; cmd.exe does not — document).
- Migrating the extension list to a single shared constant (nice-to-have; call out as follow-up).
