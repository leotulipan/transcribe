#!/usr/bin/env python3
# /// script
# dependencies = [
#   "load_dotenv",
#   "argparse",
#   "requests",
#   "datetime",
#   "pydub",
#   "loguru",
# ]
# ///

import glob
from dotenv import load_dotenv
import os
import json
import requests
from datetime import datetime
import time
import argparse
import sys
from pathlib import Path
from loguru import logger
from pydub import AudioSegment

# Import transcribe_helpers package
from transcribe_helpers import (
    # audio_processing
    check_audio_length, check_audio_format, convert_to_flac, convert_to_pcm, check_file_size,
    # utils
    setup_logger, check_transcript_exists, 
    # output_formatters
    create_srt, create_davinci_srt, create_text_file, create_word_level_srt
)
from transcribe_helpers.text_processing import standardize_word_format, process_filler_words

# Global variables
args = None
headers = None
MAX_AUDIO_LENGTH = 7200  # seconds
MAX_DAVINCI_WORDS = 500  # maximum words per subtitle block for davinci mode
FILLER_WORDS = ["äh", "ähm"]  # List of filler words to potentially remove

def get_args():
    parser = argparse.ArgumentParser(description="Audio transcription using ElevenLabs API.")
    parser.add_argument("audio_path", type=str, help="Path to the input audio file or pattern.")
    parser.add_argument("-d", "--debug", help="Debug mode", action="store_true")
    parser.add_argument("-v", "--verbose", help="Show all log messages in console", action="store_true")
    parser.add_argument("--force", help="Force re-transcription even if files exist", action="store_true")
    
    # File format options - create a mutually exclusive group
    format_group = parser.add_mutually_exclusive_group()
    format_group.add_argument("--no-convert", help="[DEPRECATED] Use --use-input instead", action="store_true")
    format_group.add_argument("--use-pcm", help="Convert to PCM WAV format (larger file size)", action="store_true")
    format_group.add_argument("--use-input", help="Use original input file without conversion (default is to convert to FLAC)", action="store_true")
    
    parser.add_argument("--keep-flac", help="Keep the generated FLAC file after processing", action="store_true")
    parser.add_argument("-l", "--language", help="Language code (ISO-639-1 or ISO-639-3)", default=None)
    parser.add_argument("-c", "--chars-per-line", type=int, default=80, help="Maximum characters per line in SRT file (default: 80). Ignored if -C/--word-srt is set.")
    parser.add_argument("-C", "--word-srt", action="store_true", help="Output SRT with each word as its own subtitle (word-level SRT, disables -c/--chars-per-line)")
    parser.add_argument("-D", "--davinci-srt", help="Export SRT for Davinci Resolve with optimized subtitle blocks (sets: chars-per-line=500, silentportions=250ms, padding-start=-125ms, remove_fillers=True, max 500 words/block). Can override with additional args.", action="store_true")
    parser.add_argument("-p", "--silentportions", type=int, help="Mark pauses longer than X milliseconds with (...) (with -D: default 250ms)", default=0)
    parser.add_argument("--padding-start", type=int, help="Milliseconds to offset word start times into preceding silence (negative=earlier, positive=later, default: 0ms, with -D: -125ms)", default=0)
    parser.add_argument("--padding-end", type=int, help="Milliseconds to offset word end times into following silence (negative=earlier, positive=later, default: 0ms)", default=0)
    parser.add_argument("--padding", type=int, help="DEPRECATED: Use --padding-end instead", default=None)
    parser.add_argument("--remove-fillers", dest="remove_fillers", help="Remove filler words, audio events, and text in parentheses (auto-enabled with -D)", action="store_true")
    parser.add_argument("--no-remove-fillers", dest="remove_fillers", help="Do not remove filler words (can override -D default)", action="store_false")
    parser.set_defaults(remove_fillers=False)
    parser.add_argument("-s", "--speaker_labels", help="Use this flag to remove speaker labels", action="store_false", default=True)
    parser.add_argument("--fps", type=float, help="Frames per second for frame-based editing (e.g., 24, 29.97, 30)", default=None)
    parser.add_argument("--fps-offset-start", type=int, help="Frames to offset from start time (default: -1, negative=earlier, positive=later)", default=-1)
    parser.add_argument("--fps-offset-end", type=int, help="Frames to offset from end time (default: 0, negative=earlier, positive=later)", default=0)
    return parser.parse_args()

def handle_error_response(response):
    """Pretty print error response from API"""
    try:
        error_data = response.json()
        detail = error_data.get('detail', {})
        status = detail.get('status', 'unknown')
        message = detail.get('message', 'No message provided')
        logger.error(f"API Error: {status} - {message}")
    except:
        logger.error(f"Raw error response: {response.text}")

def exit_with_message(message, exit_code=1):
    """
    Print a clean exit message and exit the program.
    Bypasses logger for user-friendly output regardless of debug mode.
    """
    print(f"\n{message}")
    sys.exit(exit_code)

def check_json_exists(file_dir, file_name):
    """
    Check if a JSON transcript file already exists for the given file name.
    
    Args:
        file_dir: Directory containing the file
        file_name: Base name of the file without extension
        
    Returns:
        (bool, str): (True, json_path) if JSON exists, (False, "") otherwise
    """
    # Check for ElevenLabs-specific JSON
    json_path = os.path.join(file_dir, f"{file_name}_elevenlabs.json")
    if os.path.exists(json_path):
        logger.info(f"Found ElevenLabs JSON file: {json_path}")
        return True, json_path
    
    # Check for generic JSON as fallback
    json_path = os.path.join(file_dir, f"{file_name}.json")
    if os.path.exists(json_path):
        logger.info(f"Found generic JSON file: {json_path}")
        return True, json_path
        
    return False, ""

def main():
    global args, headers
    args = get_args()

    # Setup logging
    setup_logger(args.debug, args.verbose)
    if args.debug:
        logger.info("Debug mode enabled")

    load_dotenv()
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        exit_with_message("Error: ELEVENLABS_API_KEY not found in environment variables. Please add it to your .env file.")
        
    headers = {
        "xi-api-key": api_key,
        "Accept": "application/json"
    }

    # Initialize an empty dictionary to store the files
    files_dict = {}

    # Check if args.audio_path is a directory
    if os.path.isdir(args.audio_path):
        logger.info("Directory found.")
        for root, dirs, files in os.walk(args.audio_path):
            for file in files:
                if file.endswith(('.mp3', '.wav', '.ogg', '.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4a', '.flac', '.aac', '.wma', '.aiff')):
                    files_dict[file] = os.path.join(root, file)
    # Check if args.audio_path is a file
    elif os.path.isfile(args.audio_path):
        normalized_file = os.path.normpath(args.audio_path)
        files_dict[normalized_file] = normalized_file
        logger.info("File found.")
    # Check if args.audio_path is a wildcard pattern
    elif '*' in args.audio_path or '?' in args.audio_path:
        logger.info("Wildcard pattern found.")
        for file in glob.glob(args.audio_path):
            files_dict[file] = file
    else:
        logger.error("Invalid input. Please provide a valid file, directory, or wildcard pattern.")
        return

    for file_name, file_path in files_dict.items():
        logger.info(f"Processing file: {file_name}")

        if not os.path.exists(file_path):
            logger.error(f"Audio File {file_name} does not exist!")
            continue

        full_file_name = os.path.basename(file_path)
        file_name, file_extension = os.path.splitext(full_file_name)
        file_dir = os.path.dirname(file_path)

        # Check if transcript exists
        if check_transcript_exists(file_dir, file_name) and not args.force:
            logger.info(f"Transcript for {file_name} exists! Using existing JSON to generate SRT and text files.")
            
            # Check for existing JSON file
            json_exists, json_path = check_json_exists(file_dir, file_name)
            if not json_exists:
                logger.warning(f"No JSON file found for {file_name}. Cannot regenerate outputs.")
                continue
                
            with open(json_path, 'r', encoding='utf-8') as f:
                response_data = json.load(f)
            
            # Standardize word format
            response_data['words'] = standardize_word_format(
                response_data['words'],
                'elevenlabs',
                show_pauses=args.silentportions > 0,
                silence_threshold=args.silentportions or 0
            )
            
            # Create text file
            text_file = os.path.join(file_dir, f"{file_name}.txt")
            create_text_file(response_data['words'], text_file)

            # Create SRT file
            srt_file = os.path.join(file_dir, f"{file_name}.srt")
            if args.davinci_srt:
                create_davinci_srt(
                    response_data['words'], 
                    srt_file, 
                    args.silentportions,
                    args.padding_start,
                    args.padding_end, 
                    args.fps, 
                    args.fps_offset_start, 
                    args.fps_offset_end,
                    remove_fillers=args.remove_fillers,
                    filler_words=FILLER_WORDS
                )
            elif args.word_srt:
                create_word_level_srt(
                    response_data['words'], 
                    srt_file, 
                    remove_fillers=args.remove_fillers, 
                    filler_words=FILLER_WORDS, 
                    fps=args.fps, 
                    fps_offset_start=args.fps_offset_start, 
                    fps_offset_end=args.fps_offset_end,
                    padding_start=args.padding_start,
                    padding_end=args.padding_end
                )
            else:
                create_srt(
                    response_data['words'], 
                    srt_file, 
                    args.chars_per_line, 
                    args.silentportions, 
                    args.fps, 
                    args.fps_offset_start, 
                    args.fps_offset_end,
                    args.padding_start,
                    args.padding_end,
                    remove_fillers=args.remove_fillers,
                    filler_words=FILLER_WORDS
                )
                
            logger.info(f"Transcription completed for {file_name} (from existing JSON)")
            if file_dir:
                logger.info(f"Files saved in: {file_dir}")
            continue

        # Check if JSON already exists first to skip re-encoding
        json_exists, json_path = check_json_exists(file_dir, file_name)
        if json_exists and not args.force:
            logger.info(f"JSON file for {file_name} exists! Skipping audio processing and API call.")
            with open(json_path, 'r', encoding='utf-8') as f:
                response_data = json.load(f)
                
            # Standardize word format
            response_data['words'] = standardize_word_format(
                response_data['words'],
                'elevenlabs',
                show_pauses=args.silentportions > 0,
                silence_threshold=args.silentportions or 0
            )
            
            # Create text file
            text_file = os.path.join(file_dir, f"{file_name}.txt")
            create_text_file(response_data['words'], text_file)

            # Create SRT file
            srt_file = os.path.join(file_dir, f"{file_name}.srt")
            if args.davinci_srt:
                create_davinci_srt(
                    response_data['words'], 
                    srt_file, 
                    args.silentportions,
                    args.padding_start,
                    args.padding_end, 
                    args.fps, 
                    args.fps_offset_start, 
                    args.fps_offset_end,
                    remove_fillers=args.remove_fillers,
                    filler_words=FILLER_WORDS
                )
            elif args.word_srt:
                create_word_level_srt(
                    response_data['words'], 
                    srt_file, 
                    remove_fillers=args.remove_fillers, 
                    filler_words=FILLER_WORDS, 
                    fps=args.fps, 
                    fps_offset_start=args.fps_offset_start, 
                    fps_offset_end=args.fps_offset_end,
                    padding_start=args.padding_start,
                    padding_end=args.padding_end
                )
            else:
                create_srt(
                    response_data['words'], 
                    srt_file, 
                    args.chars_per_line, 
                    args.silentportions, 
                    args.fps, 
                    args.fps_offset_start, 
                    args.fps_offset_end,
                    args.padding_start,
                    args.padding_end,
                    remove_fillers=args.remove_fillers,
                    filler_words=FILLER_WORDS
                )
                
            logger.info(f"Transcription completed for {file_name} (from existing JSON)")
            if file_dir:
                logger.info(f"Files saved in: {file_dir}")
            continue

        # Convert to appropriate format if needed
        converted_file = None
        if args.use_input:
            # Use the original file without conversion
            logger.info("Using original input file without conversion")
            file_path = file_path
        elif not args.no_convert:
            if file_extension.lower() == '.wav':
                # Check if WAV needs re-encoding
                audio = AudioSegment.from_file(file_path)
                if not check_audio_format(audio, file_extension):
                    logger.info("WAV file needs re-encoding to meet requirements")
                    converted_file = convert_to_pcm(file_path) if args.use_pcm else convert_to_flac(file_path)
                    file_path = converted_file
            elif file_extension.lower() == '.flac' and not args.use_pcm:
                # Check if FLAC needs re-encoding
                audio = AudioSegment.from_file(file_path)
                if not check_audio_format(audio, file_extension):
                    logger.info("FLAC file needs re-encoding to meet requirements")
                    converted_file = convert_to_flac(file_path)
                    file_path = converted_file
            else:
                # Convert other formats
                converted_file = convert_to_pcm(file_path) if args.use_pcm else convert_to_flac(file_path)
                file_path = converted_file
        
        # Check file size and duration
        try:
            check_file_size(file_path)
            check_audio_length(file_path, MAX_AUDIO_LENGTH)
            
            # Verify file format and readability
            try:
                probe_audio = AudioSegment.from_file(file_path)
                audio_info = f"{len(probe_audio)/1000:.1f}s, {probe_audio.channels} channel(s), {probe_audio.frame_rate}Hz, {probe_audio.sample_width*8}bit"
                logger.info(f"Audio verified: {audio_info}")
                
                # Warn about potential issues with stereo audio when not using --use-input flag
                if probe_audio.channels > 1 and not args.use_input and not args.no_convert:
                    logger.warning("Stereo audio detected. ElevenLabs works best with mono audio.")
                    if file_extension.lower() == '.flac':
                        logger.warning("Since this is a FLAC file and we're bypassing conversion, you may encounter issues.")
                        logger.warning("Consider using --use-pcm to force conversion to mono if transcription fails.")
            except Exception as e:
                error_msg = f"Cannot read audio file: {e}"
                if args.debug:
                    logger.error(error_msg)
                    logger.error("File may be corrupted or in an unsupported format")
                else:
                    exit_with_message(error_msg)
                continue
        except RuntimeError as e:
            logger.error(f"Error: {e}")
            continue

        # Change to file directory
        if file_dir:
            os.chdir(file_dir)
            logger.info(f"Working directory changed to: {file_dir}")

        # Transcribe using ElevenLabs API
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        logger.info(f"Starting ElevenLabs transcription... (File size: {file_size_mb:.2f}MB)")

        try:
            with open(file_path, 'rb') as audio_file:
                # Determine content type based on file extension and conversion options
                content_type = 'audio/wav' if args.use_pcm else 'audio/flac'
                if args.use_input:
                    if file_extension.lower() in ['.mp3']:
                        content_type = 'audio/mpeg'
                    elif file_extension.lower() in ['.wav']:
                        content_type = 'audio/wav'
                    elif file_extension.lower() in ['.flac']:
                        content_type = 'audio/flac'
                    elif file_extension.lower() in ['.ogg', '.oga']:
                        content_type = 'audio/ogg'
                    elif file_extension.lower() in ['.m4a', '.aac']:
                        content_type = 'audio/aac'
                    else:
                        content_type = 'audio/mpeg'  # Default fallback
                
                files = {
                    'file': (os.path.basename(file_path), audio_file, content_type)
                }
                data = {
                    'model_id': 'scribe_v1',
                    'language_code': args.language,
                    'tag_audio_events': 'true',
                    'num_speakers': '2' if args.speaker_labels else None,
                    'timestamps_granularity': 'word',
                    'diarize': 'true' if args.speaker_labels else 'false',
                    'file_format': 'pcm_s16le_16' if args.use_pcm else 'other'
                }
                
                # Remove None values from data
                data = {k: v for k, v in data.items() if v is not None}
                
                # Log detailed request information in debug mode
                if args.debug:
                    logger.debug(f"API Request: POST https://api.elevenlabs.io/v1/speech-to-text")
                    logger.debug(f"Headers: {json.dumps({k: v for k, v in headers.items() if k != 'xi-api-key'})}")
                    logger.debug(f"Data: {json.dumps(data)}")
                    logger.debug(f"File: {file_path} ({file_size_mb:.2f}MB, {content_type})")
                    logger.debug(f"Content-Type: {content_type}")
                
                start_time = time.time()
                response = requests.post(
                    "https://api.elevenlabs.io/v1/speech-to-text",
                    headers=headers,
                    files=files,
                    data=data
                )
                api_time = time.time() - start_time
            
            response.raise_for_status()
            response_data = response.json()
            logger.info(f"Transcription completed in {api_time:.2f}s")

            # Save JSON response
            json_file = os.path.join(file_dir, f"{file_name}_elevenlabs.json")
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, ensure_ascii=False, indent=2)
            logger.info(f"JSON response saved to {json_file}")

            # Standardize word format
            response_data['words'] = standardize_word_format(
                response_data['words'],
                'elevenlabs',
                show_pauses=args.silentportions > 0,
                silence_threshold=args.silentportions or 0
            )

            # Create text file
            text_file = os.path.join(file_dir, f"{file_name}.txt")
            create_text_file(response_data['words'], text_file)

            # Create SRT file
            srt_file = os.path.join(file_dir, f"{file_name}.srt")
            if args.davinci_srt:
                create_davinci_srt(
                    response_data['words'], 
                    srt_file, 
                    args.silentportions,
                    args.padding_start,
                    args.padding_end, 
                    args.fps, 
                    args.fps_offset_start, 
                    args.fps_offset_end,
                    remove_fillers=args.remove_fillers,
                    filler_words=FILLER_WORDS
                )
            elif args.word_srt:
                create_word_level_srt(
                    response_data['words'], 
                    srt_file, 
                    remove_fillers=args.remove_fillers, 
                    filler_words=FILLER_WORDS, 
                    fps=args.fps, 
                    fps_offset_start=args.fps_offset_start, 
                    fps_offset_end=args.fps_offset_end,
                    padding_start=args.padding_start,
                    padding_end=args.padding_end
                )
            else:
                create_srt(
                    response_data['words'], 
                    srt_file, 
                    args.chars_per_line, 
                    args.silentportions, 
                    args.fps, 
                    args.fps_offset_start, 
                    args.fps_offset_end,
                    args.padding_start,
                    args.padding_end,
                    remove_fillers=args.remove_fillers,
                    filler_words=FILLER_WORDS
                )

            logger.info(f"Transcription completed for {file_name}")
            if file_dir:
                logger.info(f"Files saved in: {file_dir}")

            # Delete temporary file unless --keep-flac is specified
            if converted_file and not args.keep_flac:
                try:
                    # Add a small delay to ensure file is released by OS
                    time.sleep(0.5)
                    converted_file.unlink(missing_ok=True)
                    logger.info(f"Deleted temporary file: {converted_file}")
                except Exception as e:
                    logger.error(f"Error deleting temporary file: {e}")

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 400:
                handle_error_response(e.response)
            elif e.response.status_code == 429:
                try:
                    error_json = e.response.json()
                    detail = error_json.get('detail', {})
                    
                    # Check if this is a system_busy error
                    if isinstance(detail, dict) and detail.get('status') == 'system_busy':
                        error_message = detail.get('message', 'The system is experiencing heavy traffic')
                        exit_with_message(f"ElevenLabs API: {error_message}\nPlease try again later.")
                    else:
                        # Regular rate limit error
                        logger.error(f"Rate limit exceeded (429 Too Many Requests). You've reached your ElevenLabs API request limit.")
                        logger.error(f"Consider waiting a while before trying again or check your subscription plan.")
                        
                        if 'detail' in error_json:
                            logger.error(f"API Message: {error_json['detail']}")
                            
                        # Check for rate limit headers
                        headers = e.response.headers
                        if 'Retry-After' in headers:
                            retry_after = headers['Retry-After']
                            logger.error(f"API suggests retrying after {retry_after} seconds")
                except Exception as parse_error:
                    if args.debug:
                        logger.error(f"Error parsing API response: {parse_error}")
                    logger.error("Rate limit exceeded. Please try again later or check your API plan.")
            elif e.response.status_code == 401:
                exit_with_message("Authentication error: Invalid API key or unauthorized access. Check your ELEVENLABS_API_KEY.")
            elif e.response.status_code == 403:
                exit_with_message("Access forbidden: Your account doesn't have permission to use this feature.")
            elif e.response.status_code == 413:
                exit_with_message(f"File too large: The audio file ({file_size_mb:.1f}MB) exceeds the API's maximum size limit.")
            elif e.response.status_code >= 500:
                exit_with_message("ElevenLabs server error. Please try again later.")
            else:
                logger.error(f"HTTP Error: {e}")
            continue
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            continue

    # Handle deprecated padding parameter
    if args.padding is not None:
        if args.padding_end == 0:  # Only use deprecated value if new param not set
            logger.warning("--padding is deprecated, use --padding-end instead")
            args.padding_end = args.padding
    
    # Apply davinci-specific defaults if the mode is enabled
    if args.davinci_srt:
        # Only apply defaults if not explicitly specified by user
        if args.silentportions == 0:
            args.silentportions = 250
        if args.chars_per_line == 80:  # If it's the default value
            args.chars_per_line = 500
        if args.padding_start == 0:
            args.padding_start = -125
        if not args.remove_fillers:  # If it wasn't explicitly set to True or False
            args.remove_fillers = True

if __name__ == '__main__':
    main() 