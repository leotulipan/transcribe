# Transcription Tool Implementation

A unified tool for transcribing audio using various APIs (AssemblyAI, ElevenLabs, Groq, OpenAI) with options for different output formats.

## Completed Tasks

- [x] AssemblyAI: Turn empty time frames between end of one word and start of next word into spacings
- [x] Add an explicit option `--show-pauses` that adds "(...)" text when pauses occur (default for `--davinci-srt`)

- [x] Create standardized word format helper function
  - [x] Implement `standardize_word_format()` in text_processing.py
  - [x] Convert AssemblyAI format (ms timestamps) to unified format (seconds)
  - [x] Ensure consistent spacing elements between words
  - [x] Add initial spacing for words not starting at timestamp 0
  - [x] Handle pause indicators based on silence threshold
  - [x] Update both scripts (elevenlabs & assemblyai) to use new helper function

- [x] Fix pause indicators in AssemblyAI SRT output
  - [x] Investigate why standardized format doesn't display pause indicators in SRT
  - [x] Create custom_export_subtitles function that uses standardized word format
  - [x] Update export_subtitles wrapper to use custom function when silence indicators needed
  - [x] Ensure consistent SRT creation with pause markers for both APIs

- [x] Have ElevenLabs and Groq use loguru like in AssemblyAI
- [x] Make sure all scripts skip re-encoding if a JSON is already present
- [x] Add the API that was used to the end of the JSON filename

- [x] Implement standardized parsers for each JSON format
  - [x] Create parser for AssemblyAI format
  - [x] Create parser for ElevenLabs format
  - [x] Create parser for Groq format
  - [x] Create unified data model for consistent access

- [x] Refactor to a unified API class
  - [x] Create base transcription class with common methods
  - [x] Implement AssemblyAI-specific implementation (submit and wait)
  - [x] Implement direct response APIs (Groq, ElevenLabs)
  - [x] Add error handling and retry logic; debug output and info with loguru

- [x] Unify all transcription scripts into a NEW central script
  - [x] Create master script with engine/model selection
  - [x] **CLI Framework:** Using Click instead of argparse for better user experience
  - [x] Created utils package with parsers, formatters, and API classes
  - [x] api (+ model where appropriate) selection via cli. defaults to groq
  - [x] check if API key is set and works before trying to access the selected api for more error robustness
  - [x] Create unified command line interface

- [x] pyinstaller and executable: preparation and setup
  - [x] Created package structure for Python package deployment
  - [x] Setup entry points for command-line usage
  - [x] Added PyInstaller configuration and build script
  
- [x] Add back OpenAI official Whisper support based on audio_transcribe.py
- [x] .env management for new users in user dir
- [x] add a switch to save the cleaned and consistent pre-processed input json as file
  - [x] the cleaned consistent json needs to have spacing elements added for some endpoints. we are only interested in the word list (start, end, text, optional: speaker)
- [x] take a json file directly as input as well not just a sound file.
  - [x] if it has a api name ie ...-assemblyai.json auto use that json format
  - [x] if not the --api parameter must be present
- [X] assemblyai error: TypeError: get_api_instance() got an unexpected keyword argument 'language'
- [x] all api, but specifically assemblyai: make sure the resutling _apiname json gets saved at the same path as the source audio/video file
- [x] make sure the default is conversion to flac for smaller file sizes and for each API check that we are then below the size limit before sending to api (this of course for all different formats we send. Size limit in TASKS.md)
- [x] fix groq json format reading to not generate empty srt e.g. from test\audio-test.mkv - note the special json format that differs from both elevenlabs and assemblyai
- [x] Re-implement proper Groq API usage for transcription
  - [x] Add chunking and FLAC conversion from groq_audio_chunking_adapted.py
  - [x] Use audio.transcriptions API endpoint rather than chat completions
  - [x] Handle long audio files by splitting into chunks
  - [x] Merge transcription chunks properly with timestamp adjustment
  - [x] run the file test\audio-test.mkv with groq and check the resulting json and srt for completeness
- [x] Add all APIs: file size limits to check before uploading, with logger.error message (assemblyai: 200MB, groq: 25MB, elevenlabs: 100MB, ...)
- [x] Properly handle Groq's decimal seconds format (S.ms) in timestamp processing for SRT generation
- [x] run `uv run .\transcribe.py --api groq -d -v --use-input ".\test\audio-test.mkv" --save-cleaned-json --word-srt`
  - [x] check that the resulting srt is word based and not just one subtitle
    - [x] Word-level SRT generation is implemented in create_srt() with srt_mode="word" but should output a srt as if normal subtitle (not .word.srt)
  - [x] check that the resulting txt file is not empty, but includes the full words "Wir testen nun das Audio Transkript und ob die Textdatei korrekt angelegt wird"
- [x] make OPENAI implementation for whisper work and saving of cleaned json and proper std name (not test\audio-test_openai_raw.json)
- [x] GROQ forgot/doesnt do FLAC conversion and chunking in default call: Fix: "File size (346.55MB) exceeds 25MB limit for groq API. Aborting"
- [x] implement proper chunking for each api with file sizes < 300mb; The proper sequence should be:
      Load the audio file
      Convert to FLAC (which usually results in smaller file size)
      Check if the FLAC file size is over the limit
      If still too large, use chunking
- [x] change file parameter cli to not be positional but e.g -f --filename
- [x] support a folder as parameter instead of just one file -F --folder
  - [x] if folder given find all audio and video files
  - [x] check the basename (no extension) and keep unique (ie keep the mp4 when also an mp3 exists) always keep the most high quality source
  - [x] loop through all files
- [x] model selection where apis support different models via cli switch with the current as default
  - [x] groq (3 whisper models)
  - [x] assemblyai (nano, ... , best)
  - [x] groq: whisper-large-v3, whisper-medium, whisper-small
  - [x] openai: whisper-1
  - [x] assemblyai: default, nano, small, medium, large, auto
- [x] assemblyai add api param disfluencies - Transcribe Filler Words, like "umm", in your media file; set to true (Now always enabled by default)
- [x] assemblyai use language_detection=True as default and only use a specific language if one is specified
- [x] check for all api's but specifically groq and assemblyai that the source json gets saved use ./test/audio-test.mkv
- [x] Invalid AssemblyAI model: whisper-large-v3, falling back to 'best': should always default to best for assemblyai and the other models according to each api default
- [x] check openai and elevenlabs API class to save the raw json
- [x] modify error handling -  "Rate limiting (429 errors) from Groq API" so when in --folder mode we stop further processing and printout the message (that tells us when we can try again how long we have to wait)
- [x] Check chunk results before merging
- [x] Track errors per file in dictionary
  - [x] Print summary at end with file paths and error types
- [x] when rate limiting; exit script right away do not try other chunks/other files
- [x] when using no --verbose or --debug we want to see what folder and file is being processed (but not details)
- [x] fix exit on chunking + 429 rate limit error: exit on first 429 - do not try to use any chunks. remove all temp transcriptions, if any
- [x] Invalid or missing Groq model: None, falling back to 'whisper-large-v3' when calling without parameters
- [x] double check language codes work as expected in all apis and if necessary write a converter (ie some api need "de" for German some deu or de_DE)
- [x] change audio_transcribe_assemblyai.bat, audio_transcribe_elevenlabs_de.bat, audio_transcribe_groq_de.bat to use the new transcribe.py instead of the old individual .py
- [x] move legacy single api audio_progressing and _transcribe scripts into a /legacy/ subdir
- [x] Fix API selection issues:
  - [x] Fix file size check using incorrect API limits (showing 25MB Groq limit even when AssemblyAI selected)
  - [x] Check all places where API name is passed to ensure correct API is used throughout the codebase
  - [x] Ensure the API selected with --api parameter is consistently used in all processing steps
  - [x] Fix the case where original file size check doesn't respect the correct API's limit (AssemblyAI: 200MB, Groq: 25MB, etc.)
- [x] davinci-srt needs to set as default: --silent-portions 250 --remove-fillers --padding-start -125 but these values need to be overriden if also given on the command line
- [x] the source json needs to ALWAYS be saved and the name of the api used needs to be added (ie FILENAME_assemblyai.json)
  - [x] if an audio/video file is given and the json for the selected API exists do not re-encode
- [x] output all relevant subtitle settings used if --debug and just before srt is generated
- [x] "DEBUG create_srt - First 5 words:" output: use official loguru throughout the whole script
- [x] 422 bug with elevenlabs
- [x] changes BASENAME_apiname.json save handling
- [x]  __main__:process_file:497 - Error processing file: name 'process_filler_words' is not defined
  - [x] test with call `uv run transcribe.py -v -d --api elevenlabs --davinci-srt --file .\test\audio-test.mkv` until error is solved
- [x] dont put api key in debug output (or mask it) e.g. with elevenlabs: 2025-05-15 09:59:48.969 | DEBUG    | utils.transcription_api:make_request:496 - Headers: {'xi-api-key': 
- [x] if --file name that is given ends in .json auto-apply/assume --use-json-input option
- [x] Characters per line: default for --davinci-srt to 500
- [x] run pause detection AFTER filler word removal and make sure pauses are output as (...)
  - [x] Debug the sequence of operations in create_output_files and create_srt_file functions
  - [x] Fix formatters.py to ensure pause detection runs after filler words are removed
  - [x] Make sure pause detection works when using JSON input with -j option
  - [x] Add debug logging to show pause lengths being processed
  - [x] Add option to output a debug file showing all word gaps and which ones were marked as pauses
  - [x] **DONE:** Added 1800 pause markers in the output SRT file with --show-pauses --silent-portions 250
  - [x] count pauses (....) with `Select-String -Pattern "\(\."  "D:\2025-05-10 und 11 - Fachfortbildung Keto Innsbruck\Signalmoleküle.srt"| Measure-Object -Line | Select-Object -ExpandProperty Lines` should be a number greater than 0
- [x] even when using _apiname.json direct to save the resulting .txt and .srt without an api name
- [x] Update filler word regex to handle ANY punctuation before and after the workd (ie chars: ",;.-" or non-word chars in general ) in process_filler_words and make sure all filler words are removed 
  - [x] make regex case insensitive
  - [x] test run the following task implementation with `uv run transcribe.py -v -d --api elevenlabs --davinci-srt --file "D:\2025-05-10 und 11 - Fachfortbildung Keto Innsbruck\Signalmoleküle_elevenlabs.json"`
  - [x] count ähm with `Select-String -Pattern 'ähm'  "D:\2025-05-10 und 11 - Fachfortbildung Keto Innsbruck\Signalmoleküle.srt"| Measure-Object -Line | Select-Object -ExpandProperty Lines` returns a number less then 7
    - **DONE:** Only 4 instances of "ähm" remain, pause detection working correctly with 1795 pause markers
- [x] **MP4 Audio Extraction for ElevenLabs** 
  - [x] Add extract_audio_from_mp4() function to audio_processing.py
  - [x] Extract audio stream from MP4 using pydub without re-encoding
  - [x] Save extracted audio as M4A format with AAC codec
  - [x] Check if extracted M4A is smaller than original MP4
  - [x] Integrate MP4 extraction in ElevenLabs API transcribe method
  - [x] Add cleanup of temporary extracted files after transcription
- [x] Fix model parameter handling bug
  - [x] In `transcribe.py`, update the model parameter handling to properly set default models when None is provided
  - [x] Ensure Groq API always gets "whisper-large-v3" as default model if none specified
  - [x] Update the debug output to show the actual model being used in all API calls
  - [x] Review model parameter handling in all API classes to ensure consistent defaults
  - [x] Add validation to prevent None being passed as model parameter to any API
- [x] Fix Groq API timestamp granularity and result handling
  - [x] Update Groq API to request word-level timestamps instead of segment-level
  - [x] Properly handle Groq API dict response in process_file function
  - [x] Ensure parse_groq_format properly handles the API response format
  - [x] Test with ./test/audio-test.mkv to verify it generates proper SRT and TXT
- [x] fix assemblyai ms timestamp detection
- [x] run test with `uv run transcribe.py --api APINAME --force --file ./test/audio-test.mkv -v -d` for all apis
  - [x] test that a _apiname.json is being created everytime
  - [x] test that a .srt and a .txt are being created each time (check the modified time)
- [x] New CLI option start-hour
  - [x] Numeric value default to 0
  - [x] with --davinci-srt set it to 1 (ie start with hour 01 instead of 00)
- [x] space after periods, commas, etc when joining subs
  - [x] add back spaces after periods and commas when joining subs ( "you,as in" => "you, as in")
  - [x] Created join_text_with_proper_spacing() helper function to handle proper spacing after punctuation
  - [x] Updated text joining logic in create_standard_srt, process_davinci_block, convert_to_srt, and create_text_file functions
  - [x] Applied fix to both transcribe_helpers/output_formatters.py and audio_transcribe/transcribe_helpers/output_formatters.py
  - [x] Updated format_transcript_with_speakers function to use proper spacing
- [x] When citing the filename in loguru, remove the path utils.transcription_api:make_request:580
  - [x] Modified file logging in utils/transcription_api.py to truncate paths and show only filename
  - [x] Updated Files info logging to use os.path.basename() for cleaner log output
- [x] **File Size Check Integration** (High Priority - Easy)
  - [x] Add file size check in `process_file()` before API call
  - [x] Use `get_api_file_size_limit()` to get ElevenLabs limit (1000MB)
  - [x] Show file size in debug output
  - [x] Stop processing if file exceeds limit after all compression attempts
- [x] **Progressive Compression Strategy** (elevenlabs)
  - [x] **Step 1:** If extracted/to encode audio file > 500MB: Apply FLAC mono compression (16kHz, mono, same as Groq)
  - [x] **Step 2:** If file still > 1000MB: Apply AAC 128kbps mono with 22.05kHz sample rate
  - [x] **Step 3:** If still > 1000MB: Error out
  - [x] Use the smallest resulting file that's under 1000MB
  - [x] Added cleanup of temporary compressed files after transcription

### join_srt_for_davinci.py
- [x] Add option to remove all pause markers ('...') from the joined SRT file
- [x] Add option to remove only pauses when the other person is speaking, with a default 500ms threshold and configurable parameter
- [x] Document the script usage and options in the README or as a docstring

- [x] Update groq models used
  DEPRECATIONS NOTICE
  This is a notice that distil-whisper-large-v3-en will be deprecated from GroqCloud.  
  We noticed you recently used distil-whisper-large-v3-en, so please be aware that if you continue setting it as your model parameter, requests will start failing August 23, 2025.
  ACTION
  migrate to the recommended replacement model,  whisper-large-v3-turbo.

## In Progress Tasks

## Future Tasks

## API Resilience Enhancement Tasks

### High Priority - Easy Implementation

- [ ] **Improve HTTP Error Handling**
  - [ ] Distinguish between retryable (500, 502, 503, 504) and non-retryable (400, 401, 403, 413) errors
  - [ ] Add specific handling for 500 Server Error vs 400 Bad Request
  - [ ] Stop retrying immediately on non-retryable errors (400, 401, 403, 413)
  - [ ] Continue retrying on server errors (500, 502, 503, 504)

- [ ] **Exponential Backoff Retry Strategy**
  - [ ] Replace fixed 5-second delay with exponential backoff
  - [ ] Start at 5 seconds, then 10s, 20s, 40s, 80s
  - [ ] Add jitter (±20% randomization) to prevent thundering herd
  - [ ] Maximum retry duration of 3 minutes total
  - [ ] Log remaining retry time for user awareness

### Medium Priority - Medium Implementation

- [ ] **Enhanced Retry Logic in `with_retry()` method**
  - [ ] Update `utils/transcription_api.py` TranscriptionAPI.with_retry()
  - [ ] Add error classification function: `is_retryable_error(status_code)`
  - [ ] Implement exponential backoff calculation
  - [ ] Add total elapsed time tracking
  - [ ] Better logging with countdown timers

- [ ] **API Health Monitoring**
  - [ ] Add simple API health check before transcription
  - [ ] Log API response times for performance monitoring  
  - [ ] Detect patterns of API instability
  - [ ] Suggest alternative APIs when one is consistently failing

### Low Priority - Advanced Features

- [ ] **Circuit Breaker Pattern**
  - [ ] Implement circuit breaker for each API
  - [ ] Open circuit after consecutive failures (e.g., 5 in a row)
  - [ ] Half-open circuit after cooldown period
  - [ ] Automatically switch to backup API when circuit is open

- [ ] **Multi-API Fallback Strategy**
  - [ ] When ElevenLabs fails consistently, auto-suggest Groq/AssemblyAI
  - [ ] Implement automatic fallback with user confirmation
  - [ ] Maintain API preference order per user

### Implementation Example:

```python
def improved_retry_strategy(self, func, max_duration_seconds=180):
    """
    Retry with exponential backoff for up to 3 minutes.
    
    Strategy:
    - 5s, 10s, 20s, 40s, 80s delays (with jitter)
    - Only retry on 500, 502, 503, 504 errors
    - Stop immediately on 400, 401, 403, 413
    - Total max duration: 3 minutes
    """
    start_time = time.time()
    attempt = 0
    base_delay = 5
    
    while time.time() - start_time < max_duration_seconds:
        try:
            return func()
        except requests.HTTPError as e:
            if not self.is_retryable_error(e.response.status_code):
                logger.error(f"Non-retryable error {e.response.status_code}, stopping retries")
                raise
                
            attempt += 1
            delay = min(base_delay * (2 ** (attempt - 1)), 80)  # Cap at 80s
            jitter = delay * 0.2 * (random.random() - 0.5)  # ±10% jitter
            actual_delay = delay + jitter
            
            remaining_time = max_duration_seconds - (time.time() - start_time)
            if remaining_time <= actual_delay:
                logger.error(f"Timeout reached, no more retries")
                raise
                
            logger.warning(f"Attempt {attempt} failed with {e.response.status_code}, retrying in {actual_delay:.1f}s (remaining: {remaining_time:.1f}s)")
            time.sleep(actual_delay)
    
    logger.error("Max retry duration exceeded")
    raise
```



- [ ] when only one _apiname json exists use this to recreate subtitles, even when no api is specified on cli
- [ ] long silence handler
  - [ ] remove long silences before converting to flac. note the exact ms position and length
  - [ ] add the silences back in when making the srt at the correct time marks as (...) for davinci, otherwise simply no subs

- [ ] check groq join algo for files > 10 min
- [ ] groq chunking timestamp debugging (save chunk json, look at join algo)
  - [x] Created debug_groq_chunking.py to analyze chunk timestamps 
  - [x] Added visualization of gaps between chunks
  - [x] Added ability to save and analyze all chunk JSON files
  - [x] Created debug_groq_chunking.bat for easy Windows execution
  - [ ] Test with longer audio files to analyze join algorithm issues
  - [ ] Identify and fix any timestamp offset issues in the merger algorithm

- [ ] gpt-4o-mini-transcribe and gpt-4o-transcribe as per https://platform.openai.com/docs/guides/speech-to-text but they have a different json format that only includes text and no timings so we cannot save srt (see https://platform.openai.com/docs/api-reference/audio/json-object or ask Context7) transcriptions

- [ ] i8n
  - [ ] come up with a robust i8n plan for all messages
  - [ ] all text in an easy definition (external file?) with english as default so we can translate and add an interface language button in the settings

- [ ] Add local-whisper/faster-whisper as local transcription option

- [ ] Streamline output formatting options
  - [ ] Implement standard SRT output format
  - [ ] Implement word-level SRT output format
  - [ ] Implement DaVinci SRT output format
  - [ ] Implement plain text output format
  - [ ] Refactor timing options (fps, padding) to be more intuitive
  - [ ] Add format-specific configuration options

- [ ] clean up. remove old unused scripts and util libaries


## Implementation Plan

The tool has been refactored to use a unified class architecture that handles all API interactions while providing a consistent interface. A common parser handles different JSON formats from various APIs, transforming them into a standardized internal format.

Output options have been streamlined with sensible defaults while maintaining flexibility for different use cases.

### Relevant Files

- `transcribe.py` - Main entry point (new central script)
- `audio_transcribe_assemblyai.py` - AssemblyAI previous implementation (existing)
- `audio_transcribe_elevenlabs.py` - ElevenLabs implementation (existing)
- `audio_transcribe_groq.py` - Groq implementation (existing)
- `audio_transcribe.py` - OpenAI Whisper implementation (existing)

#### transcribe_helpers/ (package: all helpers auto-imported)

- **audio_processing.py**
  - `check_audio_length` — Check if audio duration is within max length.
  - `check_audio_format` — Validate audio file format.
  - `convert_to_flac` — Convert audio to FLAC.
  - `convert_to_pcm` — Convert audio to PCM WAV.
  - `check_file_size` — Check if file size is under API limit.
  - `preprocess_audio` — Preprocess audio for transcription.
  - `preprocess_audio_with_ffmpeg` — Preprocess audio using ffmpeg.
  - `audio_to_base64` — Encode audio file as base64.
  - `get_api_file_size_limit` — Return max file size for given API.

- **output_formatters.py**
  - `format_time` — Format seconds to SRT time.
  - `format_time_ms` — Format ms to SRT time.
  - `retime_subtitles_fps` — Adjust timings for frame-based SRT.
  - `format_timedelta` — Format timedelta for SRT.
  - `create_srt` — Main SRT creation (standard/word/davinci).
  - `apply_intelligent_padding` — Add padding to word timings.
  - `process_davinci_block` — Write DaVinci SRT block.
  - `create_standard_srt` — Write standard SRT.
  - `create_word_level_srt` — Write word-per-line SRT.
  - `create_davinci_srt` — Write DaVinci-optimized SRT.
  - `create_text_file`