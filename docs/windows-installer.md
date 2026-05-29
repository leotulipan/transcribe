# Windows Installer Build Pipeline

This document covers everything needed to build the Windows installer for Audio
Transcribe end-to-end on a clean Win11 dev machine.

The pipeline produces a per-user installer (`dist/transcribe-setup-v<version>.exe`)
that bundles the Go CLI (`transcribe.exe`) and Fyne GUI (`transcribe-gui.exe`),
adds the CLI to PATH, creates Start Menu / optional Desktop shortcuts, and
registers a "Transcribe with..." right-click entry on common audio/video files.

GoReleaser is intentionally not used — Fyne requires CGo (a Windows host with a
C compiler), the project ships Windows-only, and the GoReleaser free tier
can't generate Windows installers. A single PowerShell orchestrator is simpler
and works identically locally and (eventually) in CI.

## TL;DR

```powershell
# One-time, on a clean machine:
winget install GoLang.Go
winget install MSYS2.MSYS2                       # for gcc (Fyne needs CGo)
winget install JRSoftware.InnoSetup              # for ISCC.exe
winget install ImageMagick.ImageMagick           # for first-run placeholder icon

# After installing MSYS2, open the MSYS2 UCRT64 shell and run:
#   pacman -Syu
#   pacman -S mingw-w64-ucrt-x86_64-gcc

# Every build:
pwsh scripts/check-installer-prereqs.ps1         # doctor — fails fast if anything missing
pwsh scripts/build-installer.ps1                 # produces dist/transcribe-setup-v<ver>.exe
pwsh scripts/verify-installer-artifacts.ps1      # assertions on the built artifacts
```

## Prerequisites

| Tool | Why | Install |
|---|---|---|
| **Go** (1.21+) | Compiles both binaries | `winget install GoLang.Go` |
| **gcc** (MinGW-w64) | Fyne uses OpenGL via CGo; needs a C compiler | `winget install MSYS2.MSYS2`, then `pacman -S mingw-w64-ucrt-x86_64-gcc` in the UCRT64 shell |
| **Inno Setup 6** | Compiles the installer (`ISCC.exe`) | `winget install JRSoftware.InnoSetup` |
| **ImageMagick** | Generates the placeholder icon on first build | `winget install ImageMagick.ImageMagick` |
| **Git** | Version is derived from the latest annotated tag | `winget install Git.Git` |
| **goversioninfo** | Embeds icon + version metadata in `.exe` files | Auto-installed by `build-installer.ps1` |

Run `pwsh scripts/check-installer-prereqs.ps1` to verify everything before
trying to build. It probes the same locations the build script uses and
reports each one as OK or MISSING with a fix hint.

### Where the C compiler is found

`build-installer.ps1` and `check-installer-prereqs.ps1` probe these locations
in order until one is found:

1. Whatever `gcc` resolves to on the current PATH
2. `C:\msys64\ucrt64\bin\gcc.exe` (MSYS2 UCRT64 — preferred)
3. `C:\msys64\mingw64\bin\gcc.exe` (MSYS2 MinGW64)
4. `C:\TDM-GCC-64\bin\gcc.exe`
5. `C:\mingw64\bin\gcc.exe`
6. `%USERPROFILE%\scoop\apps\mingw\current\bin\gcc.exe`
7. `%LOCALAPPDATA%\Programs\mingw64\bin\gcc.exe`

If found at a non-PATH location, the script prepends the containing
directory to `$env:PATH` for the duration of the build and sets
`$env:CGO_ENABLED=1`. Without this, `go build` fails inside the
`github.com/go-gl/gl` package with the cryptic "build constraints exclude
all Go files" error — that's CGo being disabled, not a Go problem.

### Where Inno Setup is found

Same pattern. Probed in order:

1. `ISCC.exe` on PATH
2. `${ProgramFiles(x86)}\Inno Setup 6\ISCC.exe`
3. `${ProgramFiles}\Inno Setup 6\ISCC.exe`
4. `%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe` (modern winget per-user install)

## Repository layout

```
assets/
  icon.svg                        existing favicon (placeholder source; can stay)
  icon-1024.png                   1024x1024 placeholder PNG (COMMITTED)
  icon.ico                        multi-res ICO 16/24/32/48/64/128/256 (COMMITTED)
installer/
  transcribe.iss                  Inno Setup script
  versioninfo.tmpl.json           goversioninfo template with ${...} placeholders
cmd/transcribe/
  main.go                         CLI entry point
  versioninfo_windows.go          //go:generate directive (windows only)
  versioninfo.json                rendered at build time (gitignored)
  resource.syso                   produced by go generate (gitignored)
cmd/transcribe-gui/
  main.go                         GUI entry point
  versioninfo_windows.go          //go:generate directive (windows only)
  versioninfo.json                rendered at build time (gitignored)
  resource.syso                   produced by go generate (gitignored)
scripts/
  build.ps1                       legacy: builds just the two binaries (not the installer)
  build-installer.ps1             full release orchestrator
  new-placeholder-icon.ps1        regenerates the placeholder icon assets
  check-installer-prereqs.ps1     doctor — verifies the toolchain
  verify-installer-artifacts.ps1  post-build assertions on dist/
dist/                             gitignored output
  transcribe.exe
  transcribe-gui.exe
  transcribe-setup-v<ver>.exe
  transcribe-setup-v<ver>.exe.sha256
```

## What happens during `build-installer.ps1`

1. **Version resolution.** From `-Version` flag, else `git describe --tags --abbrev=0` (strips leading `v`), else falls back to `0.0.0-dev`. Parses out Major/Minor/Patch for the binary FixedFileInfo block.
2. **Prereq probing.** Locates Go, gcc, goversioninfo, ISCC; installs goversioninfo on demand via `go install`.
3. **Icon bootstrap.** If `assets/icon.ico` is missing, runs `new-placeholder-icon.ps1` to regenerate it. (Committed in the repo by default, but the bootstrap path makes a fresh checkout buildable even without ImageMagick — once the file is restored.)
4. **versioninfo rendering.** Reads `installer/versioninfo.tmpl.json`, substitutes `${VERSION_*}`, `${INTERNAL_NAME}`, `${ORIGINAL_NAME}`, `${DESCRIPTION}`, writes per-binary `cmd/*/versioninfo.json`.
5. **`go generate`** runs `goversioninfo` in each cmd dir → produces `resource.syso` (icon + version metadata). The `.syso` is automatically picked up by the next `go build`.
6. **`go build`** twice:
   - CLI: `go build -ldflags "-X main.version=$Version -s -w" -o dist/transcribe.exe ./cmd/transcribe`
   - GUI: `go build -ldflags "-X main.version=$Version -H windowsgui -s -w" -o dist/transcribe-gui.exe ./cmd/transcribe-gui`
   - `-H windowsgui` makes the GUI a Windows subsystem app — no console window flashes when it launches.
7. **Inno Setup compile.** `ISCC /Qp /DAppVersion=<ver> /DSourceDir=<dist abs path> installer\transcribe.iss` → produces `dist/transcribe-setup-v<ver>.exe`.
8. **Checksum.** SHA256 written to `dist/transcribe-setup-v<ver>.exe.sha256`.

The script is idempotent — re-running it overwrites the previous artifacts.

## What the installer does at install time

- Default location: `%LOCALAPPDATA%\Programs\Transcribe` (per-user, no UAC).
- AppId: `{{4DF9FFEE-E7A0-4874-9FF5-967FAC17FB80}` — **must never change**, this is how Windows detects upgrades. Anchored in `installer/transcribe.iss`.
- Tasks (checkboxes the user can toggle on the second wizard page):
  - **Add CLI to PATH** (default on) — appends `{app}` to `HKCU\Environment\Path` via Pascal `[Code]` (`AddToUserPath` / `NeedsAddPath`). Uninstall runs `RemoveFromUserPath`, which tokenizes the path on `;` and filters out our entry — handles entries at start, middle, or end correctly.
  - **Start Menu shortcut** (default on) — launches `transcribe-gui.exe`.
  - **Desktop shortcut** (default off).
  - **Right-click "Transcribe with..."** (default on) — registers `HKCU\Software\Classes\SystemFileAssociations\<ext>\shell\Transcribe\command` for `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.aac`, `.mp4`, `.mov`, `.mkv`, `.webm`. Command: `"{app}\transcribe-gui.exe" "%1"`. `uninsdeletekey` cleans them up on uninstall.

`ChangesEnvironment=yes` and `ChangesAssociations=yes` in `[Setup]` cause
Inno to broadcast `WM_SETTINGCHANGE` at the end of install, so Explorer and
new shells pick up the PATH and shell associations immediately. (Existing
shells still need a relaunch — that's an OS limit, not an installer bug.)

## Manual smoke test after build

`verify-installer-artifacts.ps1` covers everything that can be checked without
actually running the installer. The remaining smoke test is interactive:

1. Double-click `dist\transcribe-setup-v<ver>.exe`. No UAC prompt expected.
2. Walk the wizard with all tasks checked.
3. **Open a new terminal** (PATH change is not picked up by an already-open shell). Run `transcribe --version` — should print the installed version.
4. Start Menu → "Audio Transcribe" → GUI launches with no console flash.
5. File Explorer → right-click any `.mp3` → "Transcribe with..." entry appears → launching it opens the GUI.
6. File Explorer → properties on `%LOCALAPPDATA%\Programs\Transcribe\transcribe-gui.exe` → Details tab shows version, description, company, copyright.
7. Settings → Apps → Installed apps → "Audio Transcribe" → Uninstall. After: the install dir is gone, the PATH entry is gone (check via `[Environment]::GetEnvironmentVariable('PATH','User')` in a new shell), and the right-click entry is gone.
8. Re-install the same version → upgrade flow runs without duplicate Start Menu entries or duplicate PATH entries.

## The placeholder icon

`assets/icon.ico` and `assets/icon-1024.png` ship as a deliberately rough
placeholder — a flat blue rounded square with a white "T". The intent is
that it's instantly recognizable as not-the-final-icon, so the urge to
replace it before public release is automatic.

To replace:

1. Drop a 1024x1024 PNG (transparent background, 32-bit RGBA) at `assets/icon-1024.png`.
2. Run `magick assets/icon-1024.png -define icon:auto-resize=256,128,64,48,32,24,16 assets/icon.ico`.
3. Commit both files. The next `build-installer.ps1` picks them up automatically.

To rebuild the placeholder from scratch (e.g. for a cold-start bootstrap test):

```powershell
Remove-Item assets/icon.ico, assets/icon-1024.png
pwsh scripts/new-placeholder-icon.ps1
```

`build-installer.ps1` also invokes this generator automatically when
`assets/icon.ico` is missing, so a fresh clone without ImageMagick still
builds (using the committed copy) and a clone with ImageMagick can
regenerate on demand.

## Versioning

The version comes from the latest annotated git tag of the form `v<major>.<minor>.<patch>`. To cut a new build:

```powershell
git tag -a v0.11.0 -m "Release 0.11.0"
pwsh scripts/build-installer.ps1
```

For ad-hoc test builds without tagging:

```powershell
pwsh scripts/build-installer.ps1 -Version 0.99.0-test
```

## Out of scope (intentional, follow-ups)

- **GitHub Actions release automation.** The script is written so that a workflow on `windows-latest` can simply install MSYS2 + Inno Setup + ImageMagick, then run `scripts/build-installer.ps1` and upload the artifacts via `gh release create`. No restructuring needed.
- **Code signing.** v1 ships unsigned; SmartScreen will warn until reputation accumulates. Document the click-through ("More info" → "Run anyway") in the README before public release. SignPath.io offers free signing for verified OSS projects when this becomes a priority.
- **MSI variant.** If enterprise customers need Active Directory deployment, a parallel WiX-generated `.msi` can sit alongside the `.exe` installer. Not needed for individual contributors.
- **Auto-update.** OSS users grab the new installer from GitHub Releases; no in-app updater needed.
- **Wizard graphics.** `WizardImageFile` (164x314 BMP) and `WizardSmallImageFile` (55x58 BMP) are optional Inno Setup polish; defaults look fine.

## Troubleshooting

**`build constraints exclude all Go files in github.com/go-gl/gl/v2.1/gl`**
CGo is disabled or gcc isn't on PATH. Run `check-installer-prereqs.ps1`; install MSYS2 + UCRT64 gcc if missing.

**`Inno Setup 6 not found`**
ISCC isn't at any of the probed locations. `winget install JRSoftware.InnoSetup`, then re-run. If you installed elsewhere, add its path to the probe list in `build-installer.ps1`.

**`go: command not found`**
`winget install GoLang.Go`, then start a new terminal so PATH refreshes.

**Installer runs but `transcribe --version` says "command not found" in a new terminal**
The "Add CLI to PATH" task was unchecked at install time, or the new terminal is a Windows Terminal profile that pinned its environment. Re-install with the task selected, or open a brand-new `cmd.exe` to confirm.

**Antivirus flags the installer**
Unsigned setup.exe files from unknown publishers can trip SmartScreen and some AV products. This is the expected outcome of skipping code signing in v1.
