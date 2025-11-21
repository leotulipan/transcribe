# Audio Transcribe

A powerful, easy-to-use tool for transcribing audio and video files using multiple AI transcription services. No Python knowledge required - just download, run, and transcribe!

## Quick Start

### For End Users (No Python Required)

1. **Download the Latest Release**
   - Go to [Releases](https://github.com/yourusername/audio-transcribe/releases)
   - Download `transcribe-windows-amd64.zip`
   - Extract to a folder of your choice

2. **Set Up API Keys**
   ```bash
   transcribe.exe setup
   ```
   This interactive wizard will guide you through configuring API keys for:
   - AssemblyAI
   - ElevenLabs
   - Groq
   - OpenAI

3. **Transcribe Your First File**
   ```bash
   transcribe.exe --file "path/to/your/audio.mp4" --api groq
   ```

That's it! The transcription will be saved next to your audio file.

### Using Batch Files (Even Easier!)

1. Copy `transcribe.exe` and any batch file from `batch_templates/` to the same folder
2. Drag and drop an audio/video file onto the batch file
3. Wait for transcription to complete

Example batch files:
- `transcribe_elevenlabs_de.bat` - Transcribe with ElevenLabs (German, DaVinci Resolve optimized)
- `transcribe_groq_de.bat` - Transcribe with Groq (German)
- `transcribe_assemblyai.bat` - Transcribe with AssemblyAI

## Features

- **Multiple Transcription APIs**: Choose from AssemblyAI, ElevenLabs, Groq, or OpenAI
- **Multiple Output Formats**: 
  - Plain text
  - Standard SRT subtitles
  - Word-level SRT (each word as its own subtitle)
  - DaVinci Resolve optimized SRT
- **Smart Processing**:
  - Automatic audio extraction from video files
  - Intelligent file compression to meet API limits
  - Automatic chunking for large files
  - Filler word detection and removal
  - Pause detection and marking
- **Easy to Use**:
  - Interactive setup wizard
  - Drag-and-drop batch files
  - No Python installation required (standalone executable)

## Installation

### Option 1: Pre-built Executable (Recommended)

Download the latest release from the [Releases page](https://github.com/yourusername/audio-transcribe/releases) and extract the zip file.

### Option 2: From Source (For Developers)

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/audio-transcribe.git
   cd audio-transcribe
   ```

2. Install dependencies with UV:
   ```bash
   uv sync
   ```

3. Run the tool:
   ```bash
   uv run transcribe.py --help
   ```

## Usage

### Basic Usage

```bash
transcribe.exe --file "path/to/audio.wav" --api groq
```

### Process an Entire Folder

```bash
transcribe.exe --folder "path/to/audio_files" --api groq
```

### With Language Selection

```bash
transcribe.exe --file "path/to/audio.wav" --api groq --language de
```

### DaVinci Resolve Optimized Output

```bash
transcribe.exe --file "path/to/audio.wav" --api elevenlabs --davinci-srt
```

### Advanced Options

```bash
transcribe.exe --file "path/to/audio.wav" --api elevenlabs \
  --output davinci_srt \
  --davinci-srt \
  --filler-lines \
  --silent-portions 350 \
  --padding-start -125
```

## API Keys Setup

The first time you run the tool, use the setup wizard:

```bash
transcribe.exe setup
```

This will:
- Guide you through configuring each API
- Validate your API keys
- Store keys securely in your user profile directory

**API keys are stored in:**
- Windows: `%LOCALAPPDATA%\audio_transcribe\.env`
- Linux/Mac: `~/.audio_transcribe/.env`

These files are never committed to git and are only accessible by your user account.

### Getting API Keys

- **AssemblyAI**: https://www.assemblyai.com/
- **ElevenLabs**: https://elevenlabs.io/
- **Groq**: https://groq.com/
- **OpenAI**: https://platform.openai.com/

## Output Formats

### Plain Text (`.txt`)
Simple text file with the transcription.

### Standard SRT (`.srt`)
Standard subtitle format compatible with most video players and editors.

### Word-Level SRT (`.word.srt`)
Each word appears as its own subtitle line with precise timestamps.

### DaVinci Resolve Optimized (`.srt`)
Optimized for DaVinci Resolve with:
- Filler words as separate lines (UPPERCASE)
- Pause markers `(...)` for silences ≥350ms
- Frame-accurate timing adjustments
- Customizable padding and FPS offsets

## File Size Limits

- **AssemblyAI**: Up to 200 MB per file
- **ElevenLabs**: Up to 1000 MB per file (with automatic compression)
- **Groq**: 25 MB per file (~30 minutes of audio)
- **OpenAI (Whisper)**: 25 MB per file

The tool automatically handles large files by:
- Extracting audio from video files
- Compressing audio to meet API limits
- Chunking files when necessary

## Batch File Templates

Ready-to-use batch files are available in the `batch_templates/` directory. These allow you to:

1. Drag and drop files onto the batch file
2. Automatically transcribe with pre-configured settings
3. Customize the batch files for your needs

See `batch_templates/README.md` for details.

## Command Line Options

```
Usage: transcribe.exe [OPTIONS]

Options:
  -f, --file PATH                 Audio/video file to transcribe
  -F, --folder DIRECTORY          Folder containing audio/video files
  -a, --api [assemblyai|elevenlabs|groq|openai]
                                  API to use (default: groq)
  -l, --language TEXT             Language code (ISO-639-1 or ISO-639-3)
  -o, --output [text|srt|word_srt|davinci_srt|json|all]
                                  Output format(s) (default: text,srt)
  -D, --davinci-srt               Output SRT optimized for DaVinci Resolve
  -p, --silent-portions INTEGER   Mark pauses longer than X milliseconds with (...)
  --filler-lines                   Output filler words as their own subtitle lines
  --filler-words TEXT             Custom filler words to detect
  --remove-fillers                 Remove filler words from output
  --speaker-labels                 Enable speaker labels in SRT (when available)
  --diarize                        Enable speaker diarization (ElevenLabs)
  --num-speakers INTEGER           Maximum number of speakers (1..32)
  -m, --model TEXT                Model to use (API-specific)
  -v, --verbose                    Show all log messages
  -d, --debug                      Enable debug logging
  --help                           Show this message
```

Run `transcribe.exe --help` for the complete list of options.

## Troubleshooting

### "API key not found" Error

Run the setup wizard:
```bash
transcribe.exe setup
```

### File Too Large

The tool will automatically compress or chunk large files. If you still get errors:
- Check the file size limits for your chosen API
- Try a different API with higher limits (e.g., ElevenLabs: 1000MB)

### Transcription Quality Issues

- Try a different API (each has different strengths)
- Specify the language explicitly: `--language de`
- Use a higher-quality model: `--model best` (AssemblyAI) or `--model whisper-large-v3` (Groq)

### Executable Won't Run

- Ensure you're on Windows (x64)
- Check Windows Defender isn't blocking it
- Try running from Command Prompt: `transcribe.exe --help`

## Development

For developers who want to contribute or build from source:

### Building the Executable

```bash
uv run build.py
```

The executable will be in the `dist/` directory.

### Project Structure

- `audio_transcribe/` - Main package
  - `cli.py` - Command-line interface
  - `utils/` - Utilities and API adapters
  - `transcribe_helpers/` - Transcription helpers
  - `tui/` - Interactive setup wizard
- `batch_templates/` - Ready-to-use batch files
- `legacy/` - Old scripts (archived)

### Running Tests

```bash
uv run pytest
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/audio-transcribe/issues)
- **Security**: See [SECURITY.md](SECURITY.md) for reporting vulnerabilities

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a list of changes and version history.
