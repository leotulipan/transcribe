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
 - groq ✅
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
- only one of the toggles can be active at any one time (google, openai, groq) this forces that model to be used ✅
- headers and description color in dark mode of the text is too dark: fix ✅
- input boxes are not adapted to dark mode (white background still) ✅
- recording whith ffmpeg off needs to work again. dont force ffmpeg usage
- depending on settings auto-set limit of recording length (can be overridden in settings as an input box)
  - with ffmpeg: always 10:00
  - google uncompressed: 9:30
  - openai: 2:20
- get rid of the save button next to the api key. auto save on changes
- info api key: add the links in the settings screens
  - https://aistudio.google.com/app/apikey
  - https://platform.openai.com/api-keys
  - https://console.groq.com/keys

- File size/splitting
- File Upload (Drop) over the Rec Button (change icon when drag/drop active)

- assembly ai:

toggle for eu/us To use our EU server for transcription, replace api.assemblyai.com with api.eu.assemblyai.com.
curl -X POST https://api.assemblyai.com/v2/transcript \
     -H "Authorization: <apiKey>" \
     -H "Content-Type: application/json" \
     -d '{
  "audio_url": "https://assembly.ai/wildfires.mp3",
  "audio_end_at": 280,
  "audio_start_from": 10,
  "auto_chapters": true,
  "auto_highlights": true,
  "boost_param": "high",
  "content_safety": true,
  "custom_spelling": [
    {
      "from": [
        "dicarlo"
      ],
      "to": "Decarlo"
    }
  ],
  "disfluencies": false,
  "entity_detection": true,
  "filter_profanity": true,
  "format_text": true,
  "iab_categories": true,
  "language_code": "en_us",
  "language_confidence_threshold": 0.7,
  "language_detection": true,
  "multichannel": true,
  "punctuate": true,
  "redact_pii": true,
  "redact_pii_audio": true,
  "redact_pii_audio_quality": "mp3",
  "redact_pii_policies": [
    "us_social_security_number",
    "credit_card_number"
  ],
  "redact_pii_sub": "hash",
  "sentiment_analysis": true,
  "speaker_labels": true,
  "speakers_expected": 2,
  "speech_threshold": 0.5,
  "summarization": true,
  "summary_model": "informative",
  "summary_type": "bullets",
  "topics": [
    "topics"
  ],
  "webhook_auth_header_name": "webhook-secret",
  "webhook_auth_header_value": "webhook-secret-value",
  "webhook_url": "https://your-webhook-url/path",
  "word_boost": [
    "aws",
    "azure",
    "google cloud"
  ],
  "custom_topics": true,
  "dual_channel": false
}'