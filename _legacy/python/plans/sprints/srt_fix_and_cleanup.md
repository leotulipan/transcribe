# SRT Fix & Cleanup Sprint

## ✅ Completed Tasks

### Installation & SRT Fix
- [x] **Fix uv tool installation** - Resolved by reinstalling with `uv tool install . --force`
- [x] **Test SRT generation** - Verified with `CON-136_assemblyai.json`, full SRT generated
- [x] **Verify JSON reuse** - Confirmed script skips re-transcription when JSON exists
- [x] **Clean up temporary files** - Moved `json_to_srt.py`, `combine_subtitles.py`, and test artifacts to `test/`
- [x] **Update documentation** - Updated `features.md` and created `walkthrough.md`

## 📋 Remaining Tasks (Backlog)

### CLI Refactoring (Medium Priority)
- [ ] **Fix CLI structure issue** - Change `@click.group(invoke_without_command=True)` back to `@click.command()` in `cli.py` line 418. This currently causes options like `--api` to be interpreted as commands if placed after the positional argument.
- [ ] **Remove subcommands** - Delete `@main.command()` decorators for `setup()` and `tools()` since they don't work well with `@click.command()`
- [ ] **Re-add setup command** - Create separate entry point for setup wizard if needed

### Testing (Low Priority)
- [ ] **Add tests for start_hour None handling** - Ensure regression testing for the fix

## 📝 Notes
- The SRT generation bug (TypeError with start_hour=None) is fully resolved.
- The CLI argument parsing quirk remains but is documented in `features.md`.
