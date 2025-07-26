#!/usr/bin/env python3
# /// script
# dependencies = [
#   "click",
#   "loguru",
#   "pydub",
#   "pathlib",
#   "python-dotenv",
#   "requests",
#   "assemblyai",
#   "groq",
#   "openai",
# ]
# ///

"""
Unified Audio Transcription Tool

This script provides a unified interface for transcribing audio files using
different APIs (AssemblyAI, ElevenLabs, Groq, OpenAI) with options for different output formats.
"""

import os
import glob
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
import time
from dotenv import load_dotenv
import click
import re

# Setup logging
# Try to import loguru, fallback to our mock implementation
try:
    from loguru import logger
    # Configure Loguru
    logger.add(sys.stderr, level="INFO") # Default level
except ImportError:
    # Add the parent directory to sys.path to find loguru_patch
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from loguru_patch import logger

# Import helpers
from transcribe_helpers.audio_processing import (
    check_audio_length, check_audio_format, convert_to_flac, convert_to_pcm, get_api_file_size_limit, check_file_size
)
from transcribe_helpers.utils import setup_logger
from transcribe_helpers.language_utils import get_language_code, is_language_supported
from utils.formatters import create_output_files, create_text_file, create_srt_file
from utils.parsers import TranscriptionResult, load_json_data, detect_and_parse_json, parse_json_by_api
from utils.transcription_api import get_api_instance
from transcribe_helpers.text_processing import standardize_word_format, process_filler_words
from transcribe_helpers.chunking import split_audio, transcribe_with_chunks

# Import the whole parsers module to use getattr
from utils import parsers as all_parsers # Added import

def check_json_exists(file_dir, file_name, api_name):
    """
    Check if JSON transcript files already exist for a given file.
    
    Args:
        file_dir: Directory containing the file
        file_name: Base name of the file without extension
        api_name: API name to check for specific JSON format
        
    Returns:
        Tuple of (exists, path) where exists is a boolean and path is the path to the JSON file
    """
    file_dir = Path(file_dir)
    api_json_path = file_dir / f"{file_name}_{api_name}.json"
    generic_json_path = file_dir / f"{file_name}.json"
    
    if api_json_path.exists():
        logger.info(f"Found API specific JSON: {api_json_path}")
        return True, api_json_path
    
    if generic_json_path.exists():
        logger.info(f"Found generic JSON: {generic_json_path}")
        return True, generic_json_path
    
    return False, None

def check_transcript_exists(file_path: Union[str, Path], file_name: str) -> bool:
    """
    Check if transcript files already exist for a given file.
    
    Args:
        file_path: Directory containing the file
        file_name: Base name of the file without extension
        
    Returns:
        True if transcript exists, False otherwise
    """
    transcript_path = os.path.join(file_path, f"{file_name}.txt")
    srt_path = os.path.join(file_path, f"{file_name}.srt")
    return os.path.exists(transcript_path) or os.path.exists(srt_path)


def process_file(file_path: Union[str, Path], **kwargs) -> List[str]:
    """
    Process a single audio/video file for transcription.
    
    Args:
        file_path: Path to the audio/video file
        **kwargs: Additional parameters for transcription
        
    Returns:
        List of output file paths generated
    """
    # Get parameters
    api_name = kwargs.get("api_name", "groq").lower()
    force = kwargs.get("force", False)
    save_cleaned_json = kwargs.get("save_cleaned_json", False)
    use_json_input = kwargs.get("use_json_input", False)
    check_existing = not force
    verbose = kwargs.get("verbose", False)
    debug = kwargs.get("debug", False)
    
    if debug:
        pass
    elif verbose:
        pass
    else:
        pass
        
    logger.debug(f"Processing file: {file_path}")
    logger.debug(f"API: {api_name}, Force: {force}, Save JSON: {save_cleaned_json}, Use JSON input: {use_json_input}")
    
    try:
        start_time = time.time()
        
        # Convert to Path object if it's a string
        original_input_path = Path(file_path) if isinstance(file_path, str) else file_path
        
        # Extract directory and file name components
        file_dir = original_input_path.parent
        file_name_stem = original_input_path.stem
        file_ext = original_input_path.suffix.lower()
        
        logger.debug(f"Original input path: {original_input_path}")
        logger.debug(f"File exists check: {original_input_path.exists()}")
        
        # Initialize transcription_result to None
        transcription_result = None
        
        # If using a JSON file directly as input
        if use_json_input:
            logger.info(f"Attempting to load and parse provided JSON input: {original_input_path}")
            if not original_input_path.exists():
                logger.error(f"Provided JSON input file not found: {original_input_path}")
                return []

            try:
                loaded_json_data = load_json_data(original_input_path)
                if loaded_json_data:
                    current_api_name = api_name
                    if api_name == "auto":
                        match = re.search(r'_([a-zA-Z0-9]+)(?:_raw)?(?:_cleaned)?\.json$', original_input_path.name) # Adjusted regex for flexibility
                        if match:
                            current_api_name = match.group(1).lower()
                            logger.info(f"Auto-detected API from JSON filename: {current_api_name}")
                        else:
                            logger.error(f"API is 'auto' but could not detect from filename: {original_input_path.name}. Please specify --api.")
                            return []
                    
                    # Assume loaded JSON is RAW and needs parsing
                    logger.info(f"Assuming provided JSON {original_input_path} is a raw API response for {current_api_name}.")
                    parser_func_name = f"parse_{current_api_name}_format"
                    parser_func = getattr(all_parsers, parser_func_name, None)
                    
                    if parser_func:
                        logger.info(f"Parsing loaded JSON using {parser_func_name}...")
                        transcription_result = parser_func(loaded_json_data)
                        if not isinstance(transcription_result, TranscriptionResult):
                             logger.error(f"{parser_func_name} did not return a TranscriptionResult object.")
                             transcription_result = None
                        else:
                             logger.success(f"Successfully parsed provided JSON input: {original_input_path}")
                    else:
                        logger.error(f"No parser function found for API '{current_api_name}' (expected: {parser_func_name}).")
                        transcription_result = None # Set to None if no parser
                else:
                    logger.error(f"Failed to load data from JSON input file: {original_input_path}")
                    transcription_result = None # Set to None if load fails
            except Exception as e:
                logger.error(f"Error processing JSON input file {original_input_path}: {e}")
                transcription_result = None

        elif check_existing: # Not --force and not --use-json-input
            # Try to load RAW API JSON (BASENAME_apiname.json)
            raw_json_path = file_dir / f"{file_name_stem}_{api_name}.json" 
            if raw_json_path.exists():
                logger.info(f"Found existing raw API JSON: {raw_json_path}")
                try:
                    loaded_raw_data = load_json_data(raw_json_path)
                    if loaded_raw_data:
                        parser_func_name = f"parse_{api_name}_format"
                        parser_func = getattr(all_parsers, parser_func_name, None)
                        if parser_func:
                            logger.info(f"Parsing raw API JSON ({raw_json_path}) using {parser_func_name}...")
                            transcription_result = parser_func(loaded_raw_data)
                            if not isinstance(transcription_result, TranscriptionResult):
                                 logger.error(f"{parser_func_name} did not return a TranscriptionResult object from raw data.")
                                 transcription_result = None
                            else:
                                 logger.success(f"Successfully loaded and parsed raw API JSON: {raw_json_path}")
                        else:
                            logger.error(f"No parser function found for API '{api_name}' (expected: {parser_func_name}). Cannot use raw JSON.")
                            transcription_result = None 
                    else:
                        logger.error(f"Failed to load data from raw API JSON file: {raw_json_path}")
                        transcription_result = None 
                except Exception as e:
                    logger.warning(f"Failed to load/parse raw API JSON {raw_json_path}: {e}. Will attempt API call.")
                    transcription_result = None
            else: 
                 logger.info(f"No existing raw API JSON ({raw_json_path}) found for {api_name}. Proceeding with API call.")
                 transcription_result = None 

        # If transcription_result is still None, then perform API call
        if transcription_result is None and not use_json_input : # only make API call if not from file and no existing found
            if not original_input_path.exists(): # Ensure audio file exists before API call
                logger.error(f"Audio file for transcription not found: {original_input_path}")
                return []
            current_audio_file_path = original_input_path # Set this for API call

            logger.info(f"Requesting new transcription for: {current_audio_file_path} using API: {api_name}")
            
            api_instance = get_api_instance(api_name, api_key=kwargs.get('api_key'))
            if not api_instance:
                 logger.error(f"Could not initialize API: {api_name}. Skipping file.")
                 return []
            if not api_instance.check_api_key():
                logger.error(f"Invalid or missing API key for {api_name}.")
                return []

            # File size checks and conversions (simplified, original logic was more extensive)
            # This part of the original code handled conversions and chunking decisions
            # For this refactor, we assume api_instance.transcribe handles it or the file is suitable
            # The `api_instance.transcribe` methods should internally handle chunking if needed
            # and use current_audio_file_path

            # Add file size check before API call (skip for ElevenLabs - it handles compression internally)
            if api_name.lower() != 'elevenlabs':
                from transcribe_helpers.audio_processing import get_api_file_size_limit
                max_size_mb = get_api_file_size_limit(api_name)
                current_size_mb = os.path.getsize(current_audio_file_path) / (1024 * 1024)
                logger.debug(f"File size: {current_size_mb:.2f}MB, API limit: {max_size_mb}MB")
                
                if current_size_mb > max_size_mb:
                    logger.error(f"File size ({current_size_mb:.2f}MB) exceeds {max_size_mb}MB limit for {api_name} API. Aborting")
                    return []

            # Pass all kwargs to the API instance's transcribe method
            transcribe_kwargs = kwargs.copy()
            transcribe_kwargs['original_path'] = current_audio_file_path # Pass original path for context if needed by API
            
            # model_id is specific to ElevenLabs, model for others
            if api_name == "elevenlabs":
                transcribe_kwargs['model_id'] = kwargs.get('model', 'scribe_v1') # Get model or default
                logger.debug(f"Using ElevenLabs model_id: {transcribe_kwargs['model_id']}")
            else: # for other APIs, set model with appropriate defaults
                if api_name == "groq":
                    transcribe_kwargs['model'] = kwargs.get('model', 'whisper-large-v3')
                elif api_name == "openai":
                    transcribe_kwargs['model'] = kwargs.get('model', 'whisper-1')
                elif api_name == "assemblyai":
                    transcribe_kwargs['model'] = kwargs.get('model', 'best')
                else:
                    # For any other API, still pass model if it exists
                    if 'model' in kwargs and kwargs['model'] is not None:
                        transcribe_kwargs['model'] = kwargs['model']
                
                if 'model' in transcribe_kwargs:
                    logger.debug(f"Using model: {transcribe_kwargs['model']}")


            api_call_response = api_instance.transcribe(current_audio_file_path, **transcribe_kwargs)

            # Handle API response based on its type
            logger.debug(f"API response type: {type(api_call_response)}")

            # Check if response is already a TranscriptionResult object
            if isinstance(api_call_response, TranscriptionResult):
                logger.info("API returned a TranscriptionResult object directly")
                transcription_result = api_call_response
            # Handle raw dict responses from different APIs
            elif isinstance(api_call_response, dict):
                if api_name == "assemblyai":
                    logger.info("Parsing AssemblyAI raw response...")
                    transcription_result = all_parsers.parse_assemblyai_format(api_call_response)
                elif api_name == "groq":
                    logger.info("Parsing Groq raw response...")
                    transcription_result = all_parsers.parse_groq_format(api_call_response)
                else:
                    logger.info(f"Parsing generic {api_name} raw response...")
                    # Try to use the specific parser for this API type
                    parser_func_name = f"parse_{api_name}_format"
                    parser_func = getattr(all_parsers, parser_func_name, None)
                    if parser_func:
                        transcription_result = parser_func(api_call_response)
                    else:
                        logger.error(f"No parser found for API: {api_name}")
                        transcription_result = None
            else:
                logger.error(f"{api_name} API call returned unexpected type: {type(api_call_response)}")
                transcription_result = None

        # At this point, transcription_result should be a TranscriptionResult object or None

        if not transcription_result or not isinstance(transcription_result, TranscriptionResult):
            logger.error(f"Failed to obtain a valid transcription result for {original_input_path}.")
            return [] # Exit if no valid transcription result

        logger.success(f"Successfully obtained transcription. Words count: {len(transcription_result.words)}")
        
        # Output generation using transcription_result
        output_format_types = kwargs.get("output", ["text", "srt"])
        if "all" in output_format_types:
            output_format_types = ["text", "srt", "word_srt", "davinci_srt", "json"]
        
        # Use file_name_stem from the original input for naming output files
        # current_audio_file_path might be temporary (like a converted FLAC)
        # We want output files next to the original input_path
        
        created_files_dict = create_output_files(
            result=transcription_result,
            audio_path=original_input_path, # Use original input path for naming output
            format_types=output_format_types,
            **kwargs # Pass all other formatting kwargs
        )
        output_files = list(created_files_dict.values())
        
        # Save cleaned JSON if requested (this is the parsed TranscriptionResult)
        if save_cleaned_json: # This flag might now be redundant if BASENAME_apiname.json is always saved
            cleaned_json_path = file_dir / f"{file_name_stem}_{api_name}_cleaned.json" # Distinct name
            transcription_result.save(cleaned_json_path)
            logger.info(f"Saved cleaned (parsed) JSON to: {cleaned_json_path}")
            output_files.append(str(cleaned_json_path))

        total_time = time.time() - start_time
        logger.success(f"Completed processing {original_input_path} in {total_time:.2f} seconds. Output files: {output_files}")
        return output_files
        
    except Exception as e:
        logger.error(f"Unhandled error processing file {original_input_path}: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        # Clean up potentially converted file if one was made and path exists
        # This part needs to know about converted_file_path if it was created in the API call block
        return []

def process_audio_path(audio_path: str, **kwargs) -> Tuple[int, int]:
    """
    Process an audio file or directory of audio files.
    
    Args:
        audio_path: Path to audio file or directory containing audio files
        **kwargs: Additional parameters for processing, including api_name
        
    Returns:
        Tuple of (successful_files, total_files)
    """
    path = Path(audio_path)
    api_name = kwargs.get('api_name')
    
    logger.debug(f"Processing audio path: {path} (exists: {path.exists()}, is_file: {path.is_file()}, is_dir: {path.is_dir()})")
    logger.debug(f"API name: {api_name}")
    
    # Initialize error tracking dictionary
    error_tracker = {}
    
    if not api_name:
        logger.error("No API name provided")
        return (0, 0)
    
    # Handle directory processing
    if path.is_dir():
        logger.info(f"[PROCESSING] Directory: {path}")
        
        # Define common audio and video extensions
        audio_extensions = ['.mp3', '.wav', '.flac', '.m4a', '.ogg', '.opus']
        video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.webm']
        all_media_extensions = audio_extensions + video_extensions
        
        # Find all media files in the directory
        media_files = []
        for ext in all_media_extensions:
            media_files.extend(list(path.glob(f"*{ext}")))
        
        # If no files found, try case-insensitive search with uppercase extensions
        if not media_files:
            for ext in all_media_extensions:
                media_files.extend(list(path.glob(f"*{ext.upper()}")))
        
        if not media_files:
            logger.error(f"No audio or video files found in directory: {path}")
            return (0, 0)
        
        # Group files by basename to handle duplicates with different extensions
        basename_groups = {}
        for file in media_files:
            basename = file.stem
            # Add file to its basename group or create new group
            if basename in basename_groups:
                basename_groups[basename].append(file)
            else:
                basename_groups[basename] = [file]
        
        # Process each group to select the highest quality source
        unique_files = []
        for basename, files in basename_groups.items():
            if len(files) == 1:
                # Only one file with this basename
                unique_files.append(files[0])
            else:
                # Multiple files with same basename - prioritize
                # Prioritize video over audio as it usually has better quality
                video_files = [f for f in files if f.suffix.lower() in video_extensions]
                if video_files:
                    # Pick the largest video file as it may have best quality
                    largest_file = max(video_files, key=lambda f: f.stat().st_size)
                    unique_files.append(largest_file)
                else:
                    # All audio files - prioritize FLAC > WAV > MP3 > others
                    flac_files = [f for f in files if f.suffix.lower() == '.flac']
                    if flac_files:
                        unique_files.append(flac_files[0])
                    else:
                        wav_files = [f for f in files if f.suffix.lower() == '.wav']
                        if wav_files:
                            unique_files.append(wav_files[0])
                        else:
                            # Pick the largest remaining file
                            largest_file = max(files, key=lambda f: f.stat().st_size)
                            unique_files.append(largest_file)
        
        logger.info(f"Found {len(unique_files)} unique audio/video files to process")
        
        # Process each unique file
        successful = 0
        for i, file in enumerate(unique_files):
            logger.info(f"[PROCESSING] File {i+1}/{len(unique_files)}: {file}")
            try:
                output_files = process_file(file, **kwargs)
                if output_files:
                    successful += 1
            except Exception as e:
                error_message = str(e)
                
                # Special handling for rate limit errors
                if error_message.startswith("RATE_LIMIT_ERROR:"):
                    try:
                        # Extract the JSON error info
                        error_info = json.loads(error_message.replace("RATE_LIMIT_ERROR:", "").strip())
                        
                        # Add to error tracker
                        error_tracker[str(file)] = error_info
                        
                        # Log the rate limit error
                        retry_msg = f" Retry after {error_info.get('retry_after', 'unknown')} seconds." if error_info.get('retry_after') else ""
                        logger.error(f"Rate limit exceeded for {error_info.get('api')} API.{retry_msg} Stopping folder processing.")
                        
                        # Stop processing more files
                        break
                    except json.JSONDecodeError:
                        # Fallback if error info isn't valid JSON
                        error_tracker[str(file)] = {
                            "error_type": "rate_limit_exceeded",
                            "message": error_message,
                            "file": str(file)
                        }
                        logger.error(f"Rate limit exceeded. Stopping folder processing.")
                        break
                else:
                    # Regular error tracking
                    error_tracker[str(file)] = {
                        "error_type": "processing_error",
                        "message": error_message,
                        "file": str(file)
                    }
        
        # Store error tracking data in kwargs to be accessed by the main function
        kwargs['error_tracker'] = error_tracker
        
        return (successful, len(unique_files))
    
    # Process a single file
    if path.is_file():
        logger.debug(f"Processing single file: {path}")
        
        # Auto-apply --use-json-input if file is a JSON file
        if path.suffix.lower() == '.json':
            logger.info(f"Auto-enabling JSON input mode for file: {path}")
            kwargs['use_json_input'] = True
            
            # Try to detect API from filename if not specified or if "auto"
            if api_name == "auto" or not api_name:
                # Extract API name from filename if it contains it
                filename = path.stem
                api_name_from_file = None
                if "_assemblyai" in filename:
                    api_name_from_file = "assemblyai"
                elif "_elevenlabs" in filename:
                    api_name_from_file = "elevenlabs"
                elif "_groq" in filename:
                    api_name_from_file = "groq"
                elif "_openai" in filename:
                    api_name_from_file = "openai"
                
                # If detected from filename, use that API
                if api_name_from_file:
                    logger.info(f"[PROCESSING] Detected API '{api_name_from_file}' from filename: {filename}")
                    kwargs['api_name'] = api_name_from_file
        
        try:
            output_files = process_file(path, **kwargs)
            # Store the empty error tracker in kwargs
            kwargs['error_tracker'] = error_tracker
            return (1 if output_files else 0, 1)
        except Exception as e:
            error_message = str(e)
            
            # Special handling for rate limit errors - same as in directory processing
            if error_message.startswith("RATE_LIMIT_ERROR:"):
                try:
                    error_info = json.loads(error_message.replace("RATE_LIMIT_ERROR:", "").strip())
                    error_tracker[str(path)] = error_info
                    retry_msg = f" Retry after {error_info.get('retry_after', 'unknown')} seconds." if error_info.get('retry_after') else ""
                    logger.error(f"Rate limit exceeded for {error_info.get('api')} API.{retry_msg}")
                except json.JSONDecodeError:
                    error_tracker[str(path)] = {
                        "error_type": "rate_limit_exceeded",
                        "message": error_message,
                        "file": str(path)
                    }
                    logger.error("Rate limit exceeded.")
            else:
                # Regular error tracking
                error_tracker[str(path)] = {
                    "error_type": "processing_error",
                    "message": error_message,
                    "file": str(path)
                }
            
            # Store error tracking in kwargs
            kwargs['error_tracker'] = error_tracker
            return (0, 1)
    
    logger.error(f"Path not found or invalid: {path}")
    # Store the empty error tracker in kwargs
    kwargs['error_tracker'] = error_tracker
    return (0, 0)

@click.command()
@click.option(
    "--file", "-f", 
    type=click.Path(exists=True),
    help="Audio/video file to transcribe"
)
@click.option(
    "--folder", "-F", 
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Folder containing audio/video files to transcribe"
)
@click.option(
    "--api", "-a", 
    type=click.Choice(["assemblyai", "elevenlabs", "groq", "openai"], case_sensitive=False),
    default="groq",
    help="API to use for transcription (default: groq)"
)
@click.option(
    "--language", "-l", 
    help="Language code (ISO-639-1 or ISO-639-3)",
    default=None
)
@click.option(
    "--output", "-o",
    type=click.Choice(["text", "srt", "word_srt", "davinci_srt", "json", "all"], case_sensitive=False),
    multiple=True,
    default=["text", "srt"],
    help="Output format(s) to generate (default: text,srt)"
)
@click.option(
    "--chars-per-line", "-c",
    type=int,
    default=80,
    help="Maximum characters per line in SRT file (default: 80)"
)
@click.option(
    "--word-srt", "-C",
    is_flag=True,
    help="Output SRT with each word as its own subtitle"
)
@click.option(
    "--davinci-srt", "-D",
    is_flag=True,
    help="Output SRT optimized for DaVinci Resolve"
)
@click.option(
    "--silent-portions", "-p",
    type=int,
    default=0,
    help="Mark pauses longer than X milliseconds with (...)"
)
@click.option(
    "--padding-start",
    type=int,
    default=0,
    help="Milliseconds to offset word start times (negative=earlier, positive=later)"
)
@click.option(
    "--padding-end",
    type=int,
    default=0,
    help="Milliseconds to offset word end times (negative=earlier, positive=later)"
)
@click.option(
    "--show-pauses",
    is_flag=True,
    help="Add (...) text for pauses longer than silent-portions value"
)
@click.option(
    "--remove-fillers/--no-remove-fillers",
    default=False,
    help="Remove filler words like 'äh' and 'ähm' and treat them as pauses"
)
@click.option(
    "--speaker-labels/--no-speaker-labels",
    default=True,
    help="Enable/disable speaker diarization (AssemblyAI only)"
)
@click.option(
    "--fps",
    type=float,
    help="Frames per second for frame-based editing (e.g., 24, 29.97, 30)"
)
@click.option(
    "--fps-offset-start",
    type=int,
    default=-1,
    help="Frames to offset from start time (default: -1, negative=earlier, positive=later)"
)
@click.option(
    "--fps-offset-end",
    type=int,
    default=0,
    help="Frames to offset from end time (default: 0, negative=earlier, positive=later)"
)
@click.option(
    "--use-input",
    is_flag=True,
    help="Use original input file without conversion (default is to convert to FLAC)"
)
@click.option(
    "--use-pcm",
    is_flag=True,
    help="Convert to PCM WAV format instead of FLAC (larger file size)"
)
@click.option(
    "--keep-flac",
    is_flag=True,
    help="Keep the generated FLAC file after processing"
)
@click.option(
    "--model", "-m",
    help="Model to use for transcription. API-specific defaults: groq=whisper-large-v3, openai=whisper-1, assemblyai=best. Valid options: groq=[whisper-large-v3, whisper-medium, whisper-small], openai=[whisper-1], assemblyai=[best, default, nano, small, medium, large, auto].",
    default=None  # Default will be handled per-API
)
@click.option(
    "--chunk-length",
    type=int,
    default=600,
    help="Length of each chunk in seconds for long audio (default: 600 seconds / 10 minutes)"
)
@click.option(
    "--overlap",
    type=int,
    default=10,
    help="Overlap between chunks in seconds (default: 10 seconds)"
)
@click.option(
    "--force", "-r",
    is_flag=True,
    help="Force re-transcription even if transcript exists"
)
@click.option(
    "--save-cleaned-json", "-J",
    is_flag=True,
    help="Save the cleaned and consistent pre-processed JSON file"
)
@click.option(
    "--use-json-input", "-j",
    is_flag=True,
    help="Accept JSON files as input (instead of audio files)"
)
@click.option(
    "--debug", "-d",
    is_flag=True,
    help="Enable debug logging"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show all log messages in console"
)
@click.option(
    "--start-hour",
    type=int,
    default=None,
    help="Hour offset for SRT timestamps (default: 0, with --davinci-srt: 1)"
)
def main(
    file: Optional[str],
    folder: Optional[str],
    api: str,
    language: Optional[str],
    output: List[str],
    chars_per_line: int,
    word_srt: bool,
    davinci_srt: bool,
    silent_portions: int,
    padding_start: int,
    padding_end: int,
    show_pauses: bool,
    remove_fillers: bool,
    speaker_labels: bool,
    fps: Optional[float],
    fps_offset_start: int,
    fps_offset_end: int,
    use_input: bool,
    use_pcm: bool,
    keep_flac: bool,
    model: str,
    chunk_length: int,
    overlap: int,
    force: bool,
    save_cleaned_json: bool,
    use_json_input: bool,
    debug: bool,
    verbose: bool,
    start_hour: Optional[int],
) -> None:
    """Transcribe audio/video files using various APIs."""
    # Validate input parameters
    if not file and not folder:
        click.echo("Error: Either --file or --folder must be specified.")
        sys.exit(1)
    
    if file and folder:
        click.echo("Error: Cannot specify both --file and --folder. Choose one option.")
        sys.exit(1)
    
    # Set up logging
    setup_logger(debug=debug, verbose=verbose)
    
    # Load environment variables
    load_config_from_multiple_locations()
    
    # DaVinci SRT defaults
    if davinci_srt:
        if silent_portions == 0:
            silent_portions = 250
            logger.debug("Using davinci-srt default for silent-portions: 250ms")
        if padding_start == 0:
            padding_start = -125
            logger.debug("Using davinci-srt default for padding-start: -125ms")
        if remove_fillers == False:
            remove_fillers = True
            logger.debug("Using davinci-srt default to remove filler words")
        if chars_per_line == 80:
            chars_per_line = 500
            logger.debug("Using davinci-srt default for chars-per-line: 500")
        if start_hour is None:
            start_hour = 1
            logger.debug("Using davinci-srt default for start-hour: 1")
    if start_hour is None:
        start_hour = 0
    
    # Create dictionary of parameters
    params = {
        'api_name': api,
        'language': language,
        'output': output,
        'chars_per_line': chars_per_line,
        'word_srt': word_srt,
        'davinci_srt': davinci_srt,
        'silent_portions': silent_portions,
        'padding_start': padding_start,
        'padding_end': padding_end,
        'show_pauses': show_pauses,
        'remove_fillers': remove_fillers,
        'speaker_labels': speaker_labels,
        'fps': fps,
        'fps_offset_start': fps_offset_start,
        'fps_offset_end': fps_offset_end,
        'use_input': use_input,
        'use_pcm': use_pcm,
        'keep_flac': keep_flac,
        'model': model,
        'chunk_length': chunk_length,
        'overlap': overlap,
        'force': force,
        'save_cleaned_json': save_cleaned_json,
        'use_json_input': use_json_input,
        'start_hour': start_hour,
    }
    
    # Process file or folder
    start_time = time.time()
    
    try:
        path_to_process = file if file else folder
        successful, total = process_audio_path(path_to_process, **params)
        
        processing_time = time.time() - start_time
        if total > 0:
            logger.info(f"Processed {successful}/{total} files in {processing_time:.2f} seconds")
            if successful != total:
                logger.warning(f"{total - successful} files failed processing")
        else:
            logger.error("No files were processed")
        
        # Print error summary if there are any errors
        if 'error_tracker' in params and params['error_tracker']:
            error_tracker = params['error_tracker']
            error_count = len(error_tracker)
            
            logger.info(f"\nError Summary ({error_count} files with errors):")
            
            # Group errors by type
            error_types = {}
            for file_path, error_info in error_tracker.items():
                error_type = error_info.get('error_type', 'unknown')
                if error_type not in error_types:
                    error_types[error_type] = []
                error_types[error_type].append((file_path, error_info))
            
            # Print rate limit errors first (if any)
            if 'rate_limit_exceeded' in error_types:
                rate_limit_errors = error_types['rate_limit_exceeded']
                logger.error(f"\nRate Limit Errors ({len(rate_limit_errors)} files):")
                for file_path, error_info in rate_limit_errors:
                    api = error_info.get('api', 'unknown')
                    retry_after = error_info.get('retry_after', 'unknown')
                    retry_msg = f" Retry after {retry_after} seconds." if retry_after and retry_after != 'unknown' else ""
                    logger.error(f"  {file_path}: {api} API rate limit exceeded.{retry_msg}")
                
                # Provide guidance on rate limits
                if api == 'groq':
                    logger.info("\nGroq API Rate Limits:")
                    logger.info("  Free tier: Limit of 5 requests per minute")
                    logger.info("  Pro tier: Limit of 20 requests per minute")
                    logger.info("  To continue processing, wait for the rate limit window to reset.")
            
            # Print other error types
            for error_type, errors in error_types.items():
                if error_type != 'rate_limit_exceeded':
                    logger.error(f"\n{error_type.replace('_', ' ').title()} Errors ({len(errors)} files):")
                    for file_path, error_info in errors:
                        message = error_info.get('message', 'No details available')
                        # Truncate very long messages
                        if len(message) > 150:
                            message = message[:147] + "..."
                        logger.error(f"  {file_path}: {message}")
            
            logger.info("\nTo process failed files:")
            logger.info("  1. For rate limit errors: Wait for the rate limit window to reset")
            logger.info("  2. For other errors: Check error messages and try with --force flag")
            
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        if debug:
            import traceback
            logger.debug(traceback.format_exc())
        sys.exit(1)


# --- .env management helpers ---
def load_config_from_multiple_locations():
    """
    Load configuration from multiple locations in priority order:
    1. Current working directory .env
    2. User's ~/.transcribe/.env
    3. User's ~/.env
    4. Application directory .env
    Loads only the first found .env file.
    """
    if getattr(sys, 'frozen', False):
        app_dir = Path(sys._MEIPASS)
    else:
        app_dir = Path(__file__).parent.absolute()
    config_locations = [
        Path.cwd() / '.env',
        Path.home() / '.transcribe' / '.env',
        Path.home() / '.env',
        app_dir / '.env',
    ]
    for config_path in config_locations:
        if config_path.exists():
            load_dotenv(dotenv_path=config_path, override=True)
            logger.info(f"Loaded configuration from {config_path}")
            return True
    return False

def ensure_user_config_directory():
    user_config_dir = Path.home() / '.transcribe'
    user_config_file = user_config_dir / '.env'
    if not user_config_dir.exists():
        user_config_dir.mkdir(parents=True)
        logger.info(f"Created config directory: {user_config_dir}")
    if not user_config_file.exists():
        template = """# Transcribe API Configuration\n# Add your API keys below\n\n# AssemblyAI API Key\n# ASSEMBLYAI_API_KEY=your_api_key_here\n\n# ElevenLabs API Key\n# ELEVENLABS_API_KEY=your_api_key_here\n\n# Groq API Key\n# GROQ_API_KEY=your_api_key_here\n\n# OpenAI API Key\n# OPENAI_API_KEY=your_api_key_here\n"""
        with open(user_config_file, 'w') as f:
            f.write(template)
        logger.info(f"Created template config file: {user_config_file}")
        return True
    return False


if __name__ == "__main__":
    main() 