# Current Tasks & Handoff

## 🔴 URGENT: Installation Issue (Blocking SRT Fix)

**Problem:** `uv tool install` is failing with permission errors:
```
Zugriff verweigert (os error 5) = "Access denied"
Das System kann die angegebene Datei nicht finden (os error 2) = "File not found"
```

**What's Committed:** The SRT generation bug fix is committed to git (commit 9ef5e55) but cannot be installed.

**Files Changed:**
- `audio_transcribe/cli.py` - Fixed start_hour default, updated version to 0.1.2
- `audio_transcribe/utils/formatters.py` - Fixed None handling
- `audio_transcribe/transcribe_helpers/output_formatters.py` - Added defensive None checks

**To Resolve:**
1. Close ALL running `transcribe` processes (check Task Manager)
2. Restart terminal with administrator privileges
3. Try: `uv tool uninstall audio-transcribe`
4. Then: `uv tool install .`
5. Verify: `transcribe --version` should show `0.1.2`
6. Test: `transcribe "G:\Geteilte Ablagen\Podcast\CON-136 - Peggy Dathe\output\CON-136.mp4" --api assemblyai`

## 📋 TODO List

### High Priority
- [ ] **Fix uv tool installation** (see above)
- [ ] **Test SRT generation** with existing JSON file to verify the fix works
- [ ] **Verify JSON reuse** - confirm script skips re-transcription when `CON-136_assemblyai.json` exists

### Medium Priority  
- [ ] **Fix CLI structure issue** - Change `@click.group(invoke_without_command=True)` back to `@click.command()` in `cli.py` line 418
- [ ] **Remove subcommands** - Delete `@main.command()` decorators for `setup()` and `tools()` (lines 582-595) since they don't work with `@click.command()`
- [ ] **Re-add setup command** - Create separate entry point for setup wizard if needed

### Low Priority
- [ ] Clean up temporary files (`json_to_srt.py`, `REFACTORING_PLAN.md`)
- [ ] Update documentation with JSON reuse workflow
- [ ] Add tests for start_hour None handling

## 🎯 Start Prompt for Next Session

```
Continue fixing the SRT generation bug. The code fix is committed (9ef5e55) but installation is failing with permission errors. 

Current status:
- ✅ Root cause identified: start_hour=None causing TypeError
- ✅ Fix committed: Changed defaults and added None handling
- ❌ Cannot install: Permission errors with uv tool install
- ❌ Cannot test: Need to install to verify the fix works

First, help me resolve the installation issue, then test that SRT files are generated correctly from the existing JSON file at:
G:\Geteilte Ablagen\Podcast\CON-136 - Peggy Dathe\output\CON-136_assemblyai.json
```

## 📝 Context for Next Session

**What Works:**
- JSON reuse logic (detects existing `{filename}_{api}.json` files)
- Audio optimization cascade (video → FLAC → MP3)
- Interactive TUI mode

**What's Broken:**
- SRT generation (crashes with TypeError, only writes "1")
- CLI argument parsing (treats `--api` as a command instead of option)
- Tool installation (permission errors)

**Key Files:**
- `audio_transcribe/cli.py` - Main CLI entry point
- `audio_transcribe/utils/formatters.py` - SRT file creation dispatcher
- `audio_transcribe/transcribe_helpers/output_formatters.py` - Core SRT generation logic

**Test File:**
- JSON: `G:\Geteilte Ablagen\Podcast\CON-136 - Peggy Dathe\output\CON-136_assemblyai.json` (3.1 MB, exists)
- Expected SRT: `G:\Geteilte Ablagen\Podcast\CON-136 - Peggy Dathe\output\CON-136.srt` (should have full subtitles, not just "1")