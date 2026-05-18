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

## GUI flavor

`scripts/build.ps1` produces two executables:

- `bin/transcribe.exe` — full binary. Run from a terminal for CLI/TUI/JSON
  modes; double-click to open the GUI.
- `bin/transcribe-gui.exe` — GUI-only, built with `-H windowsgui` so launching
  from Explorer never pops a console window. Identical functionality otherwise.

Distribution: ship both side-by-side in the release ZIP. Most users will pin
`transcribe-gui.exe` to their taskbar.
