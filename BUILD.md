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

# Integration tests (skipped unless API keys present)
$env:TRANSCRIBE_GROQ_KEY = "gsk_..."
go test -tags integration ./tests/integration/...
```

## Notes

- v1 ships Windows-only. macOS/Linux are v2 and require additional toolchain
  setup (Fyne CGO + macOS signing).
- The TUI/GUI delivery layers are not yet wired (Plans 3 and 4).
