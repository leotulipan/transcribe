# EDITOR Variable Diagnosis & Fix

## Findings

### Current State
- **EDITOR is not actively set anywhere.**
- `~/.claude/settings.json` — no `env` section at all
- `.claude/settings.local.json` (project) — no `env` section
- `~/.bashrc` — has the line **commented out**:
  ```bash
  # export EDITOR="'C:/Program Files/Notepad++/notepad++.exe' -wait"
  ```

### Why "Program" Appears
Classic Windows space-in-path bug. When `EDITOR=C:/Program Files/Notepad++/notepad++.exe` is set **without proper outer quoting**, the shell splits it at the space and tries to run `C:/Program` as the executable — which is why you see "Program" in the error. The rest (`Files/Notepad++/notepad++.exe`) gets passed as arguments to a non-existent binary.

The commented-out line also has incorrect quoting: `"'...'"` (outer double quotes, inner single quotes) doesn't actually quote the path for the shell — it passes the single quotes as literal characters.

### Where Claude Code Reads EDITOR
Claude Code uses `CTRL-G` to open the current file in an external editor. It reads, in order:
1. `VISUAL` env var
2. `EDITOR` env var

On Windows with Git Bash (your shell), env vars are sourced from `.bashrc` + the `env` block in `settings.json`.

### Where to Fix It (Options, best first)

**Option A — `~/.claude/settings.json` `env` block (recommended)**
Most reliable: always loaded regardless of shell, no quoting ambiguity.
```json
"env": {
  "EDITOR": "\"C:/Program Files/Notepad++/notepad++.exe\" -wait"
}
```
Or avoid the space entirely by using the short path:
```json
"env": {
  "EDITOR": "C:/Progra~1/Notepad++/notepad++.exe -wait"
}
```

**Option B — Fix `.bashrc`**
Uncomment and correct the quoting:
```bash
export EDITOR='"C:/Program Files/Notepad++/notepad++.exe" -wait'
# outer single quotes, inner double quotes — tells bash the whole thing is one string
```

**Option C — Use a path without spaces**
If Notepad++ can be installed to e.g. `C:/tools/notepad++/notepad++.exe`, no quoting needed.

### No TUI Option
Claude Code has no settings TUI for EDITOR. The only ways to set it are:
- Edit `~/.claude/settings.json` directly (the `env` block)
- Edit `~/.bashrc`
- Set it as a Windows user/system environment variable (via System Properties → Advanced → Environment Variables)

## Recommended Fix
Add an `env` section to `~/.claude/settings.json` with a properly quoted EDITOR value pointing to Notepad++.

**File to edit:** `C:\Users\leona\.claude\settings.json`
