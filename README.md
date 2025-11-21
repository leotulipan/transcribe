# Audio Transcribe

A powerful, easy-to-use tool for transcribing audio and video files using multiple AI transcription services. No Python knowledge required - just download, run, and transcribe!

## Quick Start

### For End Users (No Python Required)

1. **Download the Latest Release**
   - Go to [Releases](https://github.com/leotulipan/transcribe/releases)
   - Download `transcribe-windows-amd64.zip`
   - Extract to a folder of your choice

2. **Set Up API Keys**
   ```bash
   transcribe.exe --setup
   ```
   This interactive wizard will guide you through configuring API keys for:
   - AssemblyAI
   - ElevenLabs
   - Groq
   - OpenAI

3. **Transcribe Your First File**
   ```bash
   transcribe.exe "path/to/your/audio.mp4" --api groq
   ```

That's it! The transcription will be saved next to your audio file.

### Using Batch Files (Even Easier!)

1. Copy `transcribe.exe` and any batch file from `batch_templates/` to the same folder
2. Drag and drop an audio/video file onto the batch file
3. Wait for transcription to complete

Example batch files:
- `transcribe_elevenlabs_de.bat` - Transcribe with ElevenLabs (German, DaVinci Resolve optimized)
  - Automatically marks pauses and filler words for DaVinci Resolve Studio auto-cut
  - See "DaVinci Resolve Features" section below for details
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

Download the latest release from the [Releases page](https://github.com/leotulipan/transcribe/releases) and extract the zip file.

### Option 2: From Source (For Developers)

1. Clone the repository:
   ```bash
   git clone https://github.com/leotulipan/transcribe.git
   cd transcribe
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
transcribe.exe "path/to/audio.wav" --api groq
```

**Note**: The `--file` and `--folder` options are deprecated. Simply provide the file or folder path as a positional argument.

### Process an Entire Folder

```bash
transcribe.exe "path/to/audio_files" --api groq
```

### With Language Selection

```bash
transcribe.exe "path/to/audio.wav" --api groq --language de
```

### DaVinci Resolve Optimized Output

```bash
transcribe.exe "path/to/audio.wav" --api elevenlabs --davinci-srt
```

This creates an SRT file optimized for DaVinci Resolve Studio with pause markers that enable automatic cutting.

### Advanced DaVinci Resolve Options

```bash
transcribe.exe "path/to/audio.wav" --api elevenlabs \
  --davinci-srt \
  --filler-lines \
  --silent-portions 350 \
  --padding-start -125
```

**What this does:**
- `--davinci-srt`: Enables DaVinci Resolve optimized output
- `--filler-lines`: Outputs filler words as separate UPPERCASE subtitle lines
- `--silent-portions 350`: Marks pauses and filler words longer than 350ms as `(...)` for auto-cut
- `--padding-start -125`: Adjusts timing by -125ms (starts earlier) for frame accuracy

## API Keys Setup

The first time you run the tool, use the setup wizard:

```bash
transcribe.exe --setup
```

This will:
- Launch an interactive TUI (Text User Interface) to configure each API
- Validate your API keys
- Store keys securely in your user profile directory

**API keys are stored in:**
- Windows: `%LOCALAPPDATA%\audio_transcribe\.env`
- Linux/Mac: `~/.audio_transcribe/.env`

These files are never committed to git and are only accessible by your user account.

### Getting API Keys

- **AssemblyAI**: Register at https://www.assemblyai.com/ get the key at https://www.assemblyai.com/dashboard/api-keys
- **ElevenLabs**: Register at https://dub.link/elevenlabs get the key at https://elevenlabs.io/app/developers/api-keys
- **Groq**: Register at https://groq.com/ get the key at https://console.groq.com/keys
- **OpenAI**: Register at https://platform.openai.com/ get the key at https://platform.openai.com/settings/organization/api-keys

Note: AssemlyAI, Elevenlabs and Groq have free credits available. OpenAI afaik not anymore. You will need to load a tiny e.g. 5$ amount prepaid to try it out

## Output Formats

### Plain Text (`.txt`)
Simple text file with the transcription.

### Standard SRT (`.srt`)
Standard subtitle format compatible with most video players and editors.

### Word-Level SRT (`.word.srt`)
Each word appears as its own subtitle line with precise timestamps.

### DaVinci Resolve Optimized (`.srt`)
Optimized for DaVinci Resolve Studio with special features for automatic editing:

- **Pause Detection**: Silences and filler words longer than a specified duration (default 350ms) are marked as `(...)` in the subtitles
- **Auto-Cut Feature**: DaVinci Resolve Studio recognizes these `(...)` markers and can automatically cut the video/audio at these pause points
- **Filler Words as Separate Lines**: Filler words (like "um", "uh", "ähm") appear as their own subtitle lines in UPPERCASE, making them easy to identify and remove
- **Frame-Accurate Timing**: Adjustable timing offsets for frame-perfect synchronization
- **Customizable Padding**: Fine-tune start/end times with millisecond precision

**Example**: If you set `--silent-portions 350`, any pause or filler word longer than 350ms will become `(...)` in the SRT file. When you import this SRT into DaVinci Resolve Studio, you can use the auto-cut feature to automatically split your timeline at these pause markers, making it easy to remove unwanted silences and filler words.

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

### DaVinci Resolve Batch File

The `transcribe_elevenlabs_de.bat` file is pre-configured for DaVinci Resolve Studio workflows:

- **German language** transcription
- **DaVinci Resolve optimized** SRT output
- **Pause detection** at 350ms threshold
- **Auto-cut markers**: Pauses and filler words longer than 350ms are marked as `(...)`

When you import the resulting SRT into DaVinci Resolve Studio, you can use the auto-cut feature to automatically split your timeline at these `(...)` markers, making it easy to remove unwanted silences and filler words during editing.

See `batch_templates/README.md` for more details and customization options.

## Command Line Options

```
Usage: transcribe.exe [OPTIONS] [FILE_OR_FOLDER]

Arguments:
  [FILE_OR_FOLDER]                 Audio/video file or folder to transcribe

Options:
  -a, --api [assemblyai|elevenlabs|groq|openai]
                                  API to use (default: groq)
  -l, --language TEXT             Language code (ISO-639-1 or ISO-639-3)
  -o, --output [text|srt|word_srt|davinci_srt|json|all]
                                  Output format(s) (default: text,srt)
  -D, --davinci-srt               Output SRT optimized for DaVinci Resolve
  -p, --silent-portions INTEGER   Mark pauses longer than X milliseconds with (...)
                                  Used with --davinci-srt for auto-cut markers
  --filler-lines                   Output filler words as their own subtitle lines (UPPERCASE)
  --filler-words TEXT             Custom filler words to detect
  --remove-fillers                 Remove filler words from output
  --speaker-labels                 Enable speaker labels in SRT (when available)
  --diarize                        Enable speaker diarization (ElevenLabs)
  --num-speakers INTEGER           Maximum number of speakers (1..32)
  -m, --model TEXT                Model to use (API-specific)
  --setup                          Run interactive setup wizard for API keys
  -v, --verbose                    Show all log messages
  -d, --debug                      Enable debug logging
  --help                           Show this message
```

**Note**: The `--file` and `--folder` options are deprecated. Use positional arguments instead.

Run `transcribe.exe --help` for the complete list of options.

## Troubleshooting

### "API key not found" Error

Run the setup wizard:
```bash
transcribe.exe --setup
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
- `feature-sprints/` - Planning and documentation files

### Running Tests

```bash
uv run pytest
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/leotulipan/transcribe/issues)
- **Security**: See [SECURITY.md](SECURITY.md) for reporting vulnerabilities

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a list of changes and version history.
