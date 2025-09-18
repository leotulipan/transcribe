# Audio Transcribe

A unified tool for transcribing audio using various APIs (AssemblyAI, ElevenLabs, Groq, OpenAI) with options for different output formats.

## Features

- Support for multiple transcription APIs:
  - AssemblyAI (with speaker diarization)
  - ElevenLabs 
  - Groq (AI model-based)
  - OpenAI (Whisper)
- Various output formats:
  - Standard text output
  - SRT subtitles
  - Word-level SRT (each word as its own subtitle)
  - DaVinci Resolve optimized SRT
- Extensive configuration options:
  - Language selection
  - Silent portion detection
  - Timing adjustments (paddings and FPS-based)
  - Filler word removal
  - Multiple model options per API (from fast/lightweight to accurate/comprehensive)
- File and directory processing:
  - Process single files or entire directories
  - Intelligent file selection when multiple formats exist
  - Automatic chunking for large files

## Installation

### From Source (Development)

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd audio-transcribe
   ```

2. Create a virtual environment and install with uv:
   ```bash
   uv venv
   uv pip install -e .
   ```

### Building Executable

To build a standalone executable:

```bash
uv run build.py
```

The executable will be placed in the `./dist` directory. The build script:
- Creates a clean virtual environment
- Installs all required dependencies using UV
- Packages the application with PyInstaller
- Creates a standalone executable with all dependencies included

## Usage

### Basic Usage

```bash
transcribe --file path/to/audio.wav --api assemblyai
```

Or using the standalone executable:

```bash
dist/transcribe.exe --file path/to/audio.wav --api assemblyai
```

### Process an Entire Directory

```bash
transcribe --folder path/to/audio_files --api groq
```

### With Language Selection

```bash
transcribe --file path/to/audio.wav --api groq --language de
```

### With Custom Output Formats

```bash
transcribe --file path/to/audio.wav --api elevenlabs --output text --output srt --output davinci_srt
```

### DaVinci Resolve Optimized Output

```bash
transcribe --file path/to/audio.wav --api groq --davinci-srt
```

### Filler words as their own lines (UPPERCASE) and true pauses

Emit filler words as standalone subtitle lines in uppercase, and only mark pauses of ≥350ms with (...):

```bash
transcribe --file path/to/audio.wav --api elevenlabs --output davinci_srt --davinci-srt --filler-lines --silent-portions 350
```

Use custom filler list (repeat --filler-words):

```bash
transcribe --file path/to/audio.wav --api elevenlabs --output srt --filler-lines \
  --filler-words ähm --filler-words äh --filler-words hm --filler-words uh
```

With an existing ElevenLabs JSON transcript:

```bash
uv run transcribe.py --file "G:\Geteilte Ablagen\Podcast\CON-43 - Dr. Eva Ornella\interview-combined-audio_elevenlabs.json" \
  --use-json-input --api elevenlabs --output davinci_srt --davinci-srt --filler-lines --silent-portions 350
```

### Using Different AssemblyAI Models

AssemblyAI offers several models with different speed/accuracy tradeoffs:

```bash
# Fast transcription with lightweight model (good for short, clear audio)
transcribe --file path/to/short_meeting.mp3 --api assemblyai --model nano

# Default model for high-quality transcription (recommended for most uses)
transcribe --file path/to/important_interview.mp3 --api assemblyai --model best

# Medium-sized model (balanced speed and accuracy)
transcribe --file path/to/conference_call.mp3 --api assemblyai --model medium
```

Model options:
- `best` (default): Highest quality transcription, recommended for most use cases
- `nano`: Fastest processing, good for short/simple audio with minimal background noise
- `small`, `medium`, `large`: Different size models with increasing accuracy but longer processing time
- `auto`: Automatic model selection based on audio characteristics
- `default`: Original AssemblyAI model (not recommended for new projects)

### All Options

```
Usage: transcribe.py [OPTIONS]

  Transcribe audio/video files using various APIs.

Options:
  -f, --file PATH                 Audio/video file to transcribe
  -F, --folder DIRECTORY          Folder containing audio/video files to
                                  transcribe
  -a, --api [assemblyai|elevenlabs|groq|openai]
                                  API to use for transcription (default: groq)
  -l, --language TEXT             Language code (ISO-639-1 or ISO-639-3)
                                  (maps to language_code for ElevenLabs)
  -o, --output [text|srt|word_srt|davinci_srt|json|all]
                                  Output format(s) to generate (default:
                                  text,srt)
  -c, --chars-per-line INTEGER    Maximum characters per line in SRT file
                                  (default: 80)
  -w, --words-per-subtitle INTEGER
                                  Maximum words per subtitle block (default: 0 =
                                  disabled). Mutually exclusive with -c.
  -C, --word-srt                  Output SRT with each word as its own
                                  subtitle
  -D, --davinci-srt               Output SRT optimized for DaVinci Resolve
  -p, --silent-portions INTEGER   Mark pauses longer than X milliseconds with
                                  (...)
  --padding-start INTEGER         Milliseconds to offset word start times
                                  (negative=earlier, positive=later)
  --padding-end INTEGER           Milliseconds to offset word end times
                                  (negative=earlier, positive=later)
  --show-pauses                   Add (...) text for pauses longer than
                                  silent-portions value
  --filler-lines                  Output filler words as their own subtitle
                                  lines (uppercased). Auto-enables --show-pauses
                                  and defaults --silent-portions=350 if not set
  --filler-words TEXT             Custom filler words to detect. Repeat the flag
                                  to add more (e.g., --filler-words ähm --filler-words äh)
  --remove-fillers / --no-remove-fillers
                                  Remove filler words like 'äh' and 'ähm' and
                                  treat them as pauses
  --speaker-labels / --no-speaker-labels
                                  Enable/disable speaker labels in SRT when
                                  diarization data is available
  --diarize / --no-diarize         Enable diarization (neutral option; maps to
                                  API-specific diarization, e.g., ElevenLabs)
  --num-speakers INTEGER           Maximum number of speakers (1..32). Requires
                                  --diarize
  --fps FLOAT                     Frames per second for frame-based editing
                                  (e.g., 24, 29.97, 30)
  --fps-offset-start INTEGER      Frames to offset from start time (default:
                                  -1, negative=earlier, positive=later)
  --fps-offset-end INTEGER        Frames to offset from end time (default: 0,
                                  negative=earlier, positive=later)
  --start-hour INTEGER            Hour offset for SRT timestamps (default: 0;
                                  with --davinci-srt default is 1)
  --use-input                     Use original input file without conversion
                                  (default is to convert to FLAC)
  --use-pcm                       Convert to PCM WAV format instead of FLAC
                                  (larger file size)
  --keep-flac                     Keep the generated FLAC file after
                                  processing
  -m, --model TEXT                Model to use for transcription. API-specific
                                  options: groq=[*whisper-large-v3, whisper-
                                  medium, whisper-small], openai=[*whisper-1],
                                  assemblyai=[*best, default, nano, small, medium,
                                  large, auto]. Use nano for faster processing
                                  of short/simple audio.
  --chunk-length INTEGER          Length of each chunk in seconds for long
                                  audio (default: 600 seconds / 10 minutes)
  --overlap INTEGER               Overlap between chunks in seconds (default:
                                  10 seconds)
  -r, --force                     Force re-transcription even if transcript
                                  exists
  -J, --save-cleaned-json         Save the cleaned and consistent pre-
                                  processed JSON file
  -j, --use-json-input            Accept JSON files as input (instead of audio
                                  files)
  -d, --debug                     Enable debug logging
  -v, --verbose                   Show all log messages in console
  --help                          Show this message and exit.
```

## API Keys

You need to set up API keys for the transcription services you intend to use. Create a `.env` file with the following variables:

```
ASSEMBLYAI_API_KEY=your_assemblyai_key
ELEVENLABS_API_KEY=your_elevenlabs_key
GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key
```

### AssemblyAI Specific Features

- **Filler words (disfluencies)** like "um", "uh", etc. are always transcribed
- **Language detection** is enabled by default unless a specific language is provided
- **Speaker diarization** can be enabled/disabled with the `--speaker-labels/--no-speaker-labels` option

## License

MIT

## File Size limits

Google Gemini
Maximum File Size: 50 MB per file when using Gemini 1.5 Flash or other supported versions.

Maximum Audio Duration: 9.5 hours combined across all files in a single request.

OpenAI (Whisper and GPT-4 Audio)
Whisper: 25 MB per file.

GPT-4 Audio: No specific size limit mentioned in the provided results.

Groq
Maximum File Size: 25 MB per file (~30 minutes of audio).

AssemblyAI
File Uploads: Up to 200 MB per file for direct uploads.