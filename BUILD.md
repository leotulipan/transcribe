# Build

## Prerequisites (Windows 11)

- Go 1.22 or newer — https://go.dev/dl/
- ffmpeg + ffprobe on PATH (`winget install Gyan.FFmpeg` or `choco install ffmpeg`)

## Build

```powershell
go build -ldflags "-X main.version=dev" -o bin/transcribe.exe ./cmd/transcribe
```

Release build with embedded version:

```powershell
$ver = (git describe --tags --always)
go build -ldflags "-X main.version=$ver" -o bin/transcribe.exe ./cmd/transcribe
```

## Run tests

```powershell
go test ./...

# With race detector (slower)
go test -race ./...

# Integration tests (skipped unless the matching provider API key is present)
$env:GROQ_API_KEY        = "gsk_..."
# Or any of: OPENAI_API_KEY, ASSEMBLYAI_API_KEY, ELEVENLABS_API_KEY,
# GEMINI_API_KEY, MISTRAL_API_KEY
go test -tags integration ./...

# Or, using the helper scripts in tests/scripts/ that load from your .env:
#   source tests/scripts/load_env.sh "$HOME/.transcribe/.env" GROQ_API_KEY
#   go test -tags integration ./internal/adapters/api/groq/...
```

## Configuration

The Go binary reads configuration from three layers (later layers override earlier):

1. **User config** — `%LOCALAPPDATA%\transcribe\config.toml` on Windows
   (`~/.transcribe/config.toml` on Linux/macOS). Written by `transcribe setup`
   and by the GUI's Settings dialog.
2. **Repo-local override** — `.transcribe.toml` in the current working
   directory or any ancestor. Gitignored. Convenient for per-checkout dev keys.
3. **Environment variables** — `GROQ_API_KEY`, `OPENAI_API_KEY`,
   `ASSEMBLYAI_API_KEY`, `ELEVENLABS_API_KEY`, `GEMINI_API_KEY`,
   `MISTRAL_API_KEY`, `TRANSCRIBE_FFMPEG_PATH`. Win even over the repo-local
   file.

Note: this is **not** the same path the Python tool used
(`%LOCALAPPDATA%\audio_transcribe\.env`). The Go port intentionally uses its
own location.

The TOML schema:

```toml
default_provider = "groq"
default_language = "en"
ffmpeg_path = "C:\\Users\\you\\AppData\\Local\\Microsoft\\WinGet\\Links\\ffmpeg.exe"

[api_keys]
groq = "gsk_..."
openai = "sk-..."
# etc.
```

The GUI displays the resolved config path in the Settings dialog header.

## Notes

- v1 ships Windows-only. macOS/Linux are v2 and require additional toolchain
  setup (Fyne CGO + macOS signing).

## Reproducible build

```powershell
./scripts/build.ps1                          # version from git describe
./scripts/build.ps1 -Version v1.0.0          # explicit version
```

`./bin/transcribe.exe --version` will report the embedded version.

## Release

Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html):
`MAJOR.MINOR.PATCH`. Pre-1.0 (`0.x.y`) is fair game for incompatible changes
on each MINOR bump. Tags use a lowercase `v` prefix (`v0.9.0`, `v0.9.1`, ...).

To cut a release:

```powershell
# 1. Bump CHANGELOG.md: add a new [X.Y.Z] heading above [0.9.0], summarize
#    the changes, update the link lines at the bottom.
# 2. Commit the changelog bump.
# 3. Tag the commit:
git tag -a vX.Y.Z -m "Release X.Y.Z"
# 4. Build with the embedded version:
./scripts/build.ps1
# 5. Verify:
./bin/transcribe.exe --version   # expect: transcribe version vX.Y.Z
# 6. Push the tag (the workflow at .github/workflows/release.yml does the rest):
git push origin vX.Y.Z
```

`scripts/build.ps1` accepts `-Version` to override; when omitted it calls
`git describe --tags --always`, which yields the current tag on a tagged
commit and `vX.Y.Z-N-gSHA` on commits past the tag. Non-semver strings
(`dev`, `dev-mybranch`, etc.) are accepted with a warning so ad-hoc smoke
builds still work.

The Python tool's version (in `pyproject.toml`) is independent — it ships
its own release line. The Go binary version is authoritative for everything
under `cmd/` and `internal/`.

## GUI flavor

`scripts/build.ps1` produces two executables:

- `bin/transcribe.exe` — full binary. Run from a terminal for CLI/TUI/JSON
  modes; double-click to open the GUI.
- `bin/transcribe-gui.exe` — GUI-only, built with `-H windowsgui` so launching
  from Explorer never pops a console window. Identical functionality otherwise.

Distribution: ship both side-by-side in the release ZIP. Most users will pin
`transcribe-gui.exe` to their taskbar.
