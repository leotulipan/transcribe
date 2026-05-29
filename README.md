# Audio Transcribe

Transcribe audio and video using the best speech-to-text APIs from one tool.
Drop a file on the window, pick a provider, click **Start**. Get plain text,
standard SRT, word-level SRT, or a DaVinci Resolve–optimized SRT with pause
markers that auto-cut your timeline.

Supports **ElevenLabs Scribe**, **AssemblyAI Universal**, **Groq Whisper**,
**OpenAI Whisper**, **Google Gemini**, and **Mistral Voxtral** — switch
between them at any time, with one set of features and one set of output
formats.

## Quick start (Windows installer)

1. Download **`transcribe-setup-vX.Y.Z.exe`** from the
   [Releases page](https://github.com/leotulipan/transcribe/releases).
2. Run the installer. By default it installs to
   `%LOCALAPPDATA%\Programs\Transcribe` (per-user, no admin required) and
   adds:
   - a Start Menu shortcut,
   - a "Transcribe with…" entry in the right-click menu for `.mp3` / `.wav`
     / `.m4a` / `.flac` / `.ogg` / `.aac` / `.mp4` / `.mov` / `.mkv` /
     `.webm`,
   - the CLI on your PATH (optional, on by default).
3. Launch **Audio Transcribe** from the Start Menu. The Settings dialog
   opens automatically the first time, so you can paste your API keys.
4. Drop a file (or a folder) on the window, pick a provider and one or
   more output formats, click **Start**.

That's it. Output files are written next to the input by default, or to an
output directory you specify.

You can also right-click any supported audio/video file in Explorer and
choose **"Transcribe with…"** to open it directly in the GUI.

## Getting API keys

You need at least one API key. Click any of these links from the in-app
**Settings** dialog to register and grab your key — most providers offer
free credits or a free tier.

| Provider | Free tier | Where to get a key |
|---|---|---|
| **ElevenLabs** (Scribe) | Free credits | https://dub.link/elevenlabs |
| **AssemblyAI** (Universal) | Free credits | https://www.assemblyai.com/dashboard/signup |
| **Groq** (Whisper) | Free tier | https://console.groq.com/keys |
| **Google Gemini** | Free tier | https://aistudio.google.com/apikey |
| **Mistral** (Voxtral) | Free tier | https://console.mistral.ai/api-keys |
| **OpenAI** (Whisper) | No free credits — needs a small prepaid balance | https://platform.openai.com/settings/organization/api-keys |

Keys are stored in your user profile (`%LOCALAPPDATA%\transcribe\`) and
never leave your machine except to call the provider you choose.

## Install ffmpeg

The tool calls `ffmpeg` to extract audio from video files, convert codecs,
and compress large files to fit each provider's size limit. Without it
you can still transcribe plain `.mp3` / `.wav` / `.m4a`, but video files
and large recordings will fail.

**Recommended (one click):**

1. Open the **Settings** dialog in the GUI (the entry shows the "Install
   ffmpeg" link next to the path field).
2. Click the link — it opens [`winstall.app`](https://winstall.app/apps/Gyan.FFmpeg),
   which uses Windows' built-in `winget` to install the
   official Gyan.dev ffmpeg build.
3. After install, restart Audio Transcribe so the new PATH entry is
   picked up. Leave the **FFmpeg path** field blank and the tool will
   auto-discover it via PATH.

**Manual install:**

```powershell
winget install Gyan.FFmpeg
```

Or download a static build from <https://www.gyan.dev/ffmpeg/builds/>
and unzip it somewhere stable (e.g. `C:\Tools\ffmpeg\`). Then either:

- add `C:\Tools\ffmpeg\bin` to your `PATH` (System Properties → Environment
  Variables → Path → Edit → New), restart the GUI, leave the field blank;
- **or** paste the full path to `ffmpeg.exe` (e.g.
  `C:\Tools\ffmpeg\bin\ffmpeg.exe`) into the **FFmpeg path** field in
  Settings and click Save.

**Verify:**

```powershell
ffmpeg -version
```

If that prints a version string, the GUI will find it automatically.

## The GUI at a glance

- **Top toolbar** — Start, Cancel, Settings, About. Stays pinned so it's
  always one click away.
- **File or folder** — pick or drop. Folders are processed file by file.
- **Provider + Model** — model list updates when you switch providers;
  the **↻** button re-fetches the live list from the provider.
- **Output formats** — `text`, `srt`, `word_srt`, `davinci_srt`. Tick as
  many as you want.
- **Advanced** — collapsible sections for subtitle wrapping, diarization,
  DaVinci timing, filler words, audio pipeline, I/O & workflow, and
  provider-specific hints. Sensible defaults; only touch what you need.

## DaVinci Resolve workflow

The `davinci_srt` output is what makes this tool useful for video editors.
On top of regular subtitles it:

- marks silences and filler words longer than your threshold (default
  1500 ms) as `(...)` — DaVinci Resolve Studio's **auto-cut** treats these
  as cut points, so you can slice out the dead air in seconds;
- writes filler words (`um`, `uh`, `ähm`, …) as their own UPPERCASE
  subtitle lines so you can spot and remove them at a glance;
- supports per-millisecond start / end padding and frame-grid snapping
  (`--fps 23.976`, `--fps-offset-start -1`) for frame-accurate edits.

Workflow: transcribe with the **davinci_srt** box ticked, import the
`.davinci.srt` into Resolve, run auto-cut → done.

## CLI quick reference

The installer puts `transcribe.exe` on your PATH. Open any terminal:

```
transcribe "C:\path\to\audio.mp3"                       # default: text + srt
transcribe "C:\path\to\folder" --api elevenlabs         # batch a folder
transcribe sample.mp3 --api assemblyai --model universal-3-pro
transcribe sample.mp4 --api elevenlabs --davinci-srt --silent-portion-ms 1500
transcribe sample.mp3 --api groq --language de
transcribe --help                                       # full flag list
transcribe -V                                           # version
```

Notable flags: `-a/--api`, `-l/--language`, `-o/--output text,srt,word_srt,davinci_srt`,
`-D/--davinci-srt`, `-m/--model`, `--diarize`, `--speaker-labels`,
`--remove-fillers`, `--filler-lines`, `--padding-start`, `--padding-end`,
`--fps`, `--chunk-length`, `--force`. Run `transcribe --help` for the full
surface.

## Output formats

| File | What it is |
|---|---|
| `audio.txt` | Plain-text transcript |
| `audio.srt` | Standard SRT subtitles |
| `audio.word.srt` | Word-level SRT — one subtitle per word |
| `audio.davinci.srt` | DaVinci Resolve–optimized SRT with `(...)` pause markers and UPPERCASE filler lines |

When a `.json` sidecar exists for an input file, the tool reuses it instead
of re-transcribing — so you can re-render different formats without paying
the API a second time.

## File size limits

The tool transparently compresses, converts, and chunks large files to fit
each provider's limits:

- AssemblyAI: ~200 MB per file
- ElevenLabs: ~1000 MB per file (with compression)
- Groq, OpenAI: 25 MB per file (~30 min of audio); auto-chunked

If you hit a limit anyway, switch providers or pass `--size-threshold`
/ `--chunk-length` / `--overlap`.

## Troubleshooting

- **"API key not found"** — open the **Settings** dialog from the toolbar
  and paste a key for the provider you picked.
- **Transcription is wrong language** — provide `--language de` (or any
  ISO-639-1 code) on the CLI, or set Default language in Settings.
- **File too large** — switch to ElevenLabs (highest limit) or let the
  chunker do its thing (Groq / OpenAI auto-chunk by default).
- **Windows Defender blocks the exe** — the installer is unsigned; allow
  it once via the SmartScreen "More info" → "Run anyway" link, or run
  from a Command Prompt to confirm it's the real tool.

## Privacy

API keys are stored locally in `%LOCALAPPDATA%\transcribe\`. Audio is sent
only to the provider you select for that run; nothing is uploaded
elsewhere. The tool does not phone home or collect telemetry.

## About

By [Leonard Tulipan](https://leotulipan.at). MIT licensed — see
[LICENSE](LICENSE). Source on
[GitHub](https://github.com/leotulipan/transcribe).

---

## For developers

Audio Transcribe is written in Go (Fyne for the GUI). The Python tree was
removed in v0.10.0.

### Build from source

```powershell
git clone https://github.com/leotulipan/transcribe.git
cd transcribe

# CLI only
go build -o transcribe.exe ./cmd/transcribe

# GUI (Windows-only build tag)
go build -o transcribe-gui.exe ./cmd/transcribe-gui

# Stamp the version into the About dialog (matches installer tag):
go build -ldflags "-X main.version=v0.10.0" -o transcribe-gui.exe ./cmd/transcribe-gui

# Run the test suite
go test ./...
```

### Build the installer

```powershell
# Requires Inno Setup 6 on PATH (iscc.exe)
pwsh scripts/build-installer.ps1 -Version 0.10.0
# Output: dist/transcribe-setup-v0.10.0.exe
```

### Project layout

- `cmd/transcribe/` — CLI entry point (also dispatches to TUI / GUI)
- `cmd/transcribe-gui/` — Windows GUI entry point (Fyne)
- `internal/core/` — domain types, services, ports
- `internal/adapters/api/` — provider implementations (ElevenLabs, AssemblyAI, Groq, OpenAI, Gemini, Mistral)
- `internal/adapters/audio/` — ffmpeg wrappers, compression, chunking
- `internal/delivery/cli/`, `internal/delivery/tui/`, `internal/delivery/gui/` — three UIs sharing the same core
- `installer/transcribe.iss` — Inno Setup script
- `docs/engineering-guide.md` — architecture and conventions; read before contributing

### Contributing

Pull requests welcome. Please read
[`docs/engineering-guide.md`](docs/engineering-guide.md) first — it
documents the provider pattern, testing conventions, and shell rules. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for the basics, and
[`CHANGELOG.md`](CHANGELOG.md) for what's shipped in each release.

For security reports see [`SECURITY.md`](SECURITY.md).
