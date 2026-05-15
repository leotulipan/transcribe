# Fix: `nul` Files Being Created in Project Directory

## Context

Every time Claude Code triggers a `PermissionRequest` hook, a literal file named `nul` gets created in the current working directory (e.g., project root, `tests/acceptance/`). The hook command in `~/.claude/settings.json` uses Windows `>nul` syntax, but Claude Code executes hooks via bash (`/bin/bash.exe`), which interprets `>nul` as a redirect to a file named `nul` rather than the Windows null device.

## Root Cause

`~/.claude/settings.json`, line 71:
```
"command": "cmd /c chcp 65001 >nul && powershell -ExecutionPolicy Bypass -File %USERPROFILE%\\.claude\\claude-hook-toast.ps1"
```

Bash sees `>nul` and creates a file called `nul` in CWD. The Windows null device `NUL` only works as such inside `cmd.exe`.

## Fix

**File:** `C:/Users/leona/.claude/settings.json` (line 71)

Change:
```
cmd /c chcp 65001 >nul && powershell ...
```
To:
```
cmd /c "chcp 65001 >nul" && powershell -ExecutionPolicy Bypass -File %USERPROFILE%\.claude\claude-hook-toast.ps1
```

By quoting `"chcp 65001 >nul"`, the redirect is passed into `cmd.exe` where `nul` IS the null device. Bash only sees `cmd /c "..."` with no bash-level redirect.

## Cleanup

Delete the stale `nul` files from the repo:
- `G:\Meine Ablage\_2_Areas\Scripts\Transcribe\nul`
- `G:\Meine Ablage\_2_Areas\Scripts\Transcribe\tests\acceptance\nul`

## Verification

1. After fixing settings.json, trigger a permission request (run any Bash command requiring approval)
2. Confirm no new `nul` file appears in the project root
3. Delete existing `nul` files and confirm they don't come back
