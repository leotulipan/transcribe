# Test audio

Synthetic English speech for exercising providers and formatters. All clips
were generated with the `hexgrad/kokoro-82m` TTS model via OpenRouter's
`/api/v1/audio/speech` endpoint. They contain no real voices or personal data.

| File | Length | Content | Use it for |
|---|---|---|---|
| `speech-en-11s.mp3` | 11.3 s | One voice (`af_heart`), three sentences. Reference text in `speech-en-11s.txt`. | Word-timestamp smoke tests |
| `two-speakers-en-96s.mp3` | 96.3 s | Ten unrelated paragraphs, alternating voices `af_heart` / `am_adam` (paragraph 1 = voice A, 2 = voice B, …). Reference text in `two-speakers-en-96s.txt`, one paragraph per line. | Diarization (expect speakers to alternate per paragraph), word-count completeness (250 words) |
| `tts-bad-duration-header.mp3` | ~10.7 s of audio | Same text as `speech-en-11s`, raw TTS MP3 output (24 kHz MPEG-2). | Regression sample: MAI-Transcribe-2 stops after ~3 s on this file. Re-encoding through ffmpeg fixes it. |

`../short-sample.mp3` contains no speech; providers return an empty transcript.

## Long files

Long files are not committed (10–15 MB each). Build them by looping a clip:

```sh
ffmpeg -stream_loop 140 -i speech-en-11s.mp3 -ac 1 -b:a 48k long-27min.mp3
```

Looped audio is identical audio repeated, which some models treat specially:
MAI-Transcribe-2 dropped about 43% of the repetitions, while Gemini 3.5
Transcribe kept all of them. Use a looped file to test size and duration
limits, not completeness. To test completeness, use `two-speakers-en-96s.mp3`.

## Regenerating

POST to `https://openrouter.ai/api/v1/audio/speech` with JSON
`{"model":"hexgrad/kokoro-82m","input":"<text>","voice":"af_heart","response_format":"pcm"}`.
The response is raw 16-bit mono PCM at 24 kHz. Wrap it with
`ffmpeg -f s16le -ar 24000 -ac 1 -i in.pcm -b:a 64k out.mp3`.
The `mp3` response format produces files like `tts-bad-duration-header.mp3`,
so request `pcm` and encode the MP3 yourself.
