# Batch Templates and Launch Assets

This directory contains ready-to-use batch files, shortcuts, and icons for launching the transcription tool.

## Batch Files

Copy any of these `.bat` files to your desired location and customize as needed:

- **transcribe_elevenlabs_youtube.bat** - Transcribe with ElevenLabs API
- **transcribe_elevenlabs_de.bat** - Transcribe with ElevenLabs API (German, DaVinci Resolve optimized)
- **transcribe_assemblyai.bat** - Transcribe with AssemblyAI API
- **transcribe_groq_de.bat** - Transcribe with Groq API (German)

### Usage

1. Copy `transcribe.exe` to the same directory as the batch file
2. Drag and drop an audio/video file onto the batch file, OR
3. Run from command line: `transcribe_elevenlabs_de.bat "path\to\file.mp4"`

### Customization

Edit the batch files to change:
- API selection (`--api elevenlabs|groq|assemblyai|openai`)
- Language (`--language de|en|...`)
- Output format (`--davinci-srt`, `--output srt`, etc.)
- Other options (see `transcribe.exe --help`)

## Shortcuts (.lnk)

Windows shortcuts for quick access:
- **Transcribe Elevenlabs.lnk** - Launch with ElevenLabs
- **Transcribe AssemblyAI.lnk** - Launch with AssemblyAI
- **Transcribe Groq DE.lnk** - Launch with Groq (German)

To use: Right-click → Properties → Update the "Target" path to point to your `transcribe.exe` location.

## Icons

- **elevenlabs.ico** - ElevenLabs icon
- **assemblyai.ico** - AssemblyAI icon
- **groq.ico** - Groq icon

Use these icons when creating your own shortcuts or batch files.

