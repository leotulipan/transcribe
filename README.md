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
- ✅ show token (in/out) usage next to each file (size, duration, tokens)
- ✅ recording history: max 10 entries (auto-delete the oldest)
- ✅ Settings wheel should toggle the display on/off - not just the X
- ✅ Dark mode (System/Light/Dark) switch in settings
- ✅ remove token in/out from settings screen (only keep them for each recording when we have the info)
- ✅ toggle switch to activate ffmpeg encode (on by default)
- ✅ Support different Transcription APIs
  - ✅ toggle switch in front of Gemini API
  - ✅ toggle switch + api key for OpenAI
    - ✅ Additional settings: dropdown for model (whisper, gpt-4o-audio-preview chat completion, gpt-4o-mini-audio-preview chat completion)
    - ✅ language: auto or select from dropdown (for whisper)
 - groq
    - whisper-large-v3-turbo or distil-whisper-large-v3-en (english only) or whisper-large-v3
    - groq uses the openai api syntax
    ```curl https://api.groq.com/openai/v1/audio/transcriptions \
  -H "Authorization: bearer ${GROQ_API_KEY}" \
  -F "file=@./sample_audio.m4a" \
  -F model=whisper-large-v3-turbo \
  -F temperature=0 \
  -F response_format=text \
  -F timestamp_granularities=["word"] \
  -F language=en```
  - other apis coming later (plan for them)
- info api key: add the links
  - https://aistudio.google.com/app/apikey
  - https://platform.openai.com/api-keys
  - https://console.groq.com/keys
- only one of the toggles can be active at any one time (google, openai, groq) this forces that model to be used
- headers and description color in dark mode of the text is too dark: fix
- input boxes are not adapted to dark mode (white background still)
- depending on settings auto-set limit of recording length (can be overridden in settings as an input box)
  - with ffmpeg: always 10:00
  - google uncompressed: 9:30
  - openai: 2:20

- File size/splitting
- File Upload (Drop) over the Rec Button (change icon when drag/drop active)