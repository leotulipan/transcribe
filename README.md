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

# Todo

- ✅ show file size in MB an MM:SS
- ✅ Download button for audio
- show token (in/out) usage next to each file (size, duration, tokens)
- recording history: max 10 entries (auto-delete the oldest)
- Settings wheel should toggle the display on/off - not just the X
- Dark mode (System/Light/Dark) switch in settings
- toggle switch to activate ffmpeg encode (on by default)
- Support different Transcription APIs
  - toggle switch in front of Gemini API
  - toggle switch + api key for OpenAI
    - Additional settings: dropdown for model (whisper, gpt-4o-audio-preview chat completion, gpt-4o-mini-audio-preview chat completion)
    - language: auto or select from dropdown (for whisper)
  - other apis coming later (plan for them)
- depending on settings auto-set limit of recording length (can be overridden in settings as an input box)
  - with ffmpeg: always 10:00
  - google uncompressed: 9:30
  - openai: 2:20

- File size/splitting
- File Upload (Drop) over the Rec Button (change icon when drag/drop active)