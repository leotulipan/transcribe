# Windows Installer for Audio Transcribe (Go CLI + Fyne GUI)

## Context

The Go port reached feature-parity with the legacy Python build at v0.10.0 and now ships two
working binaries: `transcribe.exe` (CLI) and `transcribe-gui.exe` (Fyne GUI). The Python
PyInstaller release is being retired. We need a Windows installer that bundles both
binaries, installs per-user without admin, adds the CLI to PATH, creates Start Menu /
optional Desktop shortcuts, and registers a "Transcribe with..." right-click entry for
common audio/video files.

v1 scope is a **local, repeatable build on Win11**. GitHub Actions automation is a
follow-up. The build must be end-to-end runnable from a fresh checkout — including a
generated placeholder icon — so nothing blocks on a hand-supplied asset.

## Decision Summary

| Choice | Decision | Rationale |
|---|---|---|
| Build orchestration | Single `scripts/build-installer.ps1` | One source of truth; runs identically local & CI. Skip GoReleaser — CGo/Fyne forces a Windows host anyway, free tier can't build the installer. |
| Icon / version metadata | `goversioninfo` → `.syso` | Standard Go pattern; embeds icon + Explorer "Details" tab metadata. |
| Installer | Inno Setup 6 | Per-user friendly, scriptable, free, well-documented Pascal-ish syntax. |
| Install scope | Per-user (`{localappdata}\Programs\Transcribe`) | No UAC; modern OSS default (VS Code, GitHub CLI). |
| Code signing | Skip v1, document SmartScreen warning | Revisit when reputation matters. |
| Icon for v1 | ImageMagick-generated placeholder, committed to repo | Unblocks build; design pass is a separate follow-up. |

## File Layout

```
assets/
  icon.svg                     # existing favicon.svg as seed (kept for designer follow-up)
  icon-1024.png                # generated placeholder (committed)
  icon.ico                     # generated multi-res ICO (committed)
installer/
  transcribe.iss               # Inno Setup script
  versioninfo.tmpl.json        # template, version filled at build time
cmd/transcribe/
  main.go                      # add //go:generate directive
  versioninfo.json             # generated, gitignored
  resource.syso                # generated, gitignored
cmd/transcribe-gui/
  main.go                      # add //go:generate directive
  versioninfo.json             # generated, gitignored
  resource.syso                # generated, gitignored
scripts/
  build.ps1                    # existing — left alone
  build-installer.ps1          # NEW — full release orchestrator
  new-placeholder-icon.ps1     # NEW — idempotent icon generator
dist/                          # gitignored
  transcribe.exe
  transcribe-gui.exe
  transcribe-setup-v<ver>.exe
  transcribe-setup-v<ver>.exe.sha256
```

## Critical Files

### `scripts/build-installer.ps1` (new — orchestrator)

Flow:

1. **Version**: arg override OR `git describe --tags --abbrev=0` (strip leading `v`); fallback `0.0.0-dev`.
2. **Prereqs check**:
   - `go` on PATH (fail loudly if missing).
   - `goversioninfo`: install via `go install github.com/josephspurrier/goversioninfo/cmd/goversioninfo@latest` if not present in `GOBIN`/`GOPATH\bin`.
   - `ISCC.exe`: search PATH, then `${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe`. Fail with `winget install JRSoftware.InnoSetup` hint if missing.
   - `magick.exe`: search PATH; only needed if `assets/icon.ico` is missing.
3. **Icon bootstrap**: if `assets/icon.ico` doesn't exist, invoke `scripts/new-placeholder-icon.ps1`.
4. **Version metadata**: render `cmd/transcribe/versioninfo.json` and `cmd/transcribe-gui/versioninfo.json` from `installer/versioninfo.tmpl.json` with per-binary substitutions.
5. **Resource compilation**: `go generate ./cmd/transcribe ./cmd/transcribe-gui` → produces `resource.syso` next to each `main.go`.
6. **Build**:
   - `go build -ldflags "-X main.version=$Version -s -w" -o dist/transcribe.exe ./cmd/transcribe`
   - `go build -ldflags "-X main.version=$Version -H windowsgui -s -w" -o dist/transcribe-gui.exe ./cmd/transcribe-gui`
7. **Sanity check**: `Get-Item dist/*.exe | ForEach-Object { $_.VersionInfo }` — confirm icon + version embedded.
8. **Inno Setup**: `& $ISCC /Qp /DAppVersion=$Version /DSourceDir=$(Resolve-Path dist) installer\transcribe.iss`
9. **Checksum**: `Get-FileHash dist\transcribe-setup-v$Version.exe -Algorithm SHA256` → write `.sha256` sidecar.
10. Print summary table (paths, sizes, version).

### `scripts/new-placeholder-icon.ps1` (new — placeholder generator)

Self-contained. Generates a clearly-placeholder icon end-to-end so the build never blocks.

- Use ImageMagick to render a 1024×1024 PNG: rounded square in a distinctive color (e.g. `#3B82F6` blue) with a stark white "T" centered. Clearly identifiable as "this is a placeholder, replace before public release."
- Convert to multi-res ICO: `magick icon-1024.png -define icon:auto-resize=256,128,64,48,32,24,16 icon.ico`
- Outputs: `assets/icon-1024.png`, `assets/icon.ico`
- Idempotent: silently overwrites.

### `installer/transcribe.iss` (new — Inno Setup script)

Highlights:

```
[Setup]
AppId={{...fixed-GUID...}     ; generate once, never change (upgrade detection)
AppName=Audio Transcribe
AppVersion={#AppVersion}
AppPublisher=Leonard Tulipan
AppPublisherURL=https://github.com/leotulipan/transcribe
DefaultDirName={localappdata}\Programs\Transcribe
DefaultGroupName=Audio Transcribe
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#SourceDir}
OutputBaseFilename=transcribe-setup-v{#AppVersion}
SetupIconFile=..\assets\icon.ico
WizardStyle=modern
ChangesEnvironment=yes      ; needed for PATH change broadcast
ChangesAssociations=yes     ; needed for shell context menu broadcast
UninstallDisplayIcon={app}\transcribe-gui.exe

[Tasks]
Name: "addtopath";     Description: "Add CLI to PATH (transcribe command)"; GroupDescription: "CLI integration:"
Name: "startmenuicon"; Description: "Start Menu shortcut";                  GroupDescription: "Shortcuts:"
Name: "desktopicon";   Description: "Desktop shortcut";                     GroupDescription: "Shortcuts:"; Flags: unchecked
Name: "shellcontext";  Description: "Add ""Transcribe with..."" to right-click menu for audio/video files"; GroupDescription: "Integration:"

[Files]
Source: "{#SourceDir}\transcribe.exe";     DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\transcribe-gui.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\README.md";                    DestDir: "{app}"; Flags: isreadme
Source: "..\LICENSE";                      DestDir: "{app}"

[Icons]
Name: "{group}\Audio Transcribe";       Filename: "{app}\transcribe-gui.exe"; Tasks: startmenuicon
Name: "{group}\Uninstall";              Filename: "{uninstallexe}";           Tasks: startmenuicon
Name: "{userdesktop}\Audio Transcribe"; Filename: "{app}\transcribe-gui.exe"; Tasks: desktopicon

[Registry]
; Shell context menu for media files (HKCU per-user — no admin needed).
; Repeat for: .mp3, .mp4, .wav, .m4a, .flac, .mov, .mkv, .webm, .ogg, .aac
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.mp3\shell\Transcribe";          ValueType: string; ValueData: "Transcribe with..."; Tasks: shellcontext; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.mp3\shell\Transcribe\command";  ValueType: string; ValueData: """{app}\transcribe-gui.exe"" ""%1"""; Tasks: shellcontext

[Code]
// NeedsAddPath: returns True if {app} not already in user PATH (avoids duplicates on reinstall).
// AddToUserPath / RemoveFromUserPath: read HKCU\Environment\Path, modify, write back, broadcast WM_SETTINGCHANGE.
// Called from CurStepChanged(ssPostInstall) when addtopath task is selected, and from CurUninstallStepChanged.
```

PATH manipulation is done in `[Code]` rather than `[Registry]` because direct `[Registry]` PATH writes don't handle the "append to existing value, avoid duplicates, broadcast change" semantics safely.

### `installer/versioninfo.tmpl.json` (new — template)

Standard `goversioninfo` JSON with placeholders. Build script substitutes:

| Placeholder | CLI value | GUI value |
|---|---|---|
| `${VERSION}` | from git tag | from git tag |
| `${INTERNAL_NAME}` | `transcribe` | `transcribe-gui` |
| `${ORIGINAL_NAME}` | `transcribe.exe` | `transcribe-gui.exe` |
| `${DESCRIPTION}` | `Audio Transcribe (CLI)` | `Audio Transcribe` |

Both share: `CompanyName=Leonard Tulipan`, `ProductName=Audio Transcribe`, `LegalCopyright=MIT — github.com/leotulipan/transcribe`, `IconPath=../../assets/icon.ico`.

### `cmd/transcribe/main.go` + `cmd/transcribe-gui/main.go` (modify)

Add at top, after the `package main` line:

```go
//go:generate goversioninfo -64 -platform-specific=false
```

### `.gitignore` (modify)

Add:
```
dist/
cmd/transcribe/resource.syso
cmd/transcribe/versioninfo.json
cmd/transcribe-gui/resource.syso
cmd/transcribe-gui/versioninfo.json
```

`assets/icon.ico` and `assets/icon-1024.png` are **committed** so contributors without
ImageMagick can build out of the box. The placeholder generator is only invoked when the
file is missing.

## Verification (end-to-end smoke test)

1. `Remove-Item dist -Recurse -Force -ErrorAction SilentlyContinue` → clean slate.
2. `pwsh scripts/build-installer.ps1` → exits 0; produces `dist/transcribe-setup-v<ver>.exe`.
3. **Bootstrap test** (one-time): delete `assets/icon.ico` and `assets/icon-1024.png`, re-run the build, confirm placeholder regenerates and build succeeds; then `git checkout assets/icon.ico assets/icon-1024.png` to restore the committed copy.
4. Run the installer in current Windows session — no UAC prompt; finishes per-user.
5. **New terminal** (PATH change requires a fresh shell) → `transcribe --version` prints expected version.
6. Start Menu → "Audio Transcribe" launches GUI.
7. Right-click any `.mp3` → "Transcribe with..." entry → launches GUI (file argument behavior is GUI's own concern; not part of this plan).
8. File Explorer → `%LOCALAPPDATA%\Programs\Transcribe\transcribe-gui.exe` → Properties → Details: shows version, description, copyright, placeholder icon.
9. Add/Remove Programs → uninstall → PATH entry gone, registry context menu gone, files gone.
10. Re-run installer (same version) → upgrade flow runs cleanly, no duplicate Start Menu entries, no double PATH entry.

## Out of Scope (follow-ups)

- GitHub Actions workflow (`.github/workflows/release.yml`) that runs `build-installer.ps1` on tag push and attaches artifacts via `gh release create`. Easy to add once local works.
- Real branded icon — design pass; the placeholder ships v1.
- Inno Setup `WizardImageFile` (164×314 BMP) and `WizardSmallImageFile` (55×58 BMP) — defaults fine.
- Code signing — revisit when needed.
- Removing the old Python build path from `.github/workflows/release.yml` (touched in same release PR but logically separate).
- MSI / WiX variant.
- Auto-update.

## Prerequisites on the Dev Machine

- Go (already required).
- Inno Setup 6: `winget install JRSoftware.InnoSetup`.
- ImageMagick — confirmed installed; only invoked on first build when icon is absent.
- `goversioninfo` — installed automatically by the build script.
