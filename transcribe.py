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

# Assume helper functions like check_json_exists are defined later or imported
# from utils.file_utils import check_json_exists # Example

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
    api_name = kwargs.get("api_name", "groq").lower()  # Ensure we use api_name from kwargs
    force = kwargs.get("force", False)
    save_cleaned_json = kwargs.get("save_cleaned_json", False)
    use_input_json = kwargs.get("use_input_json", False)
    check_existing = not force
    verbose = kwargs.get("verbose", False)
    debug = kwargs.get("debug", False)
    
    # Configure logger level based on verbosity
    if debug:
        logger.level = "DEBUG"
    elif verbose:
        logger.level = "INFO"
    else:
        logger.level = "SUCCESS"  # Show only success and higher
        
    # Show which file is being processed (always, even in non-verbose mode)
    print(f"Processing file: {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return []
    
    # Convert to Path object for easier handling
    file_path = Path(file_path)
    original_file_path = file_path  # Keep for reference
    output_files = []
    
    # Standardize path handling
    file_dir = file_path.parent
    file_name = file_path.stem
    file_ext = file_path.suffix.lower()
    
    # Start processing time
    start_time = time.time()
    
    try:
        # If instructed to use input as JSON, load it directly
        raw_json_data = None
        if use_input_json:
            # Load direct from the provided JSON file
            logger.info(f"Loading provided JSON input: {file_path}")
            
            # Determine API from filename if possible
            if api_name == "auto":
                # Try to auto-detect API from filename
                api_match = re.search(r'-(assemblyai|groq|openai|elevenlabs)\.json$', str(file_path), re.IGNORECASE)
                if api_match:
                    detected_api = api_match.group(1).lower()
                    logger.info(f"Auto-detected API from filename: {detected_api}")
                    api_name = detected_api
                    kwargs["api_name"] = api_name
                else:
                    logger.error("Could not auto-detect API from filename and no API specified with --api")
                    return []
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_json_data = json.load(f)
                logger.info(f"Successfully loaded JSON data from: {file_path}")
            except Exception as e:
                logger.error(f"Failed to load JSON file: {e}")
                raw_json_data = None
        
        # Check if we already have a JSON file for this audio with the correct API
        elif check_existing:
            # Possible JSON filenames to check:
            # - basename-api.json (e.g., audio-assemblyai.json)
            # - basename.json (legacy)
            # - basename_api_raw.json (legacy)
            
            api_json_path = file_dir / f"{file_name}-{api_name}.json"
            legacy_json_path = file_dir / f"{file_name}.json"
            raw_json_path = file_dir / f"{file_name}_{api_name}_raw.json"  # Legacy format
            
            json_paths = [api_json_path, legacy_json_path, raw_json_path]
            
            for json_path in json_paths:
                if json_path.exists():
                    logger.info(f"Found existing JSON: {json_path}")
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            raw_json_data = json.load(f)
                        
                        # Verify this is actually for the right API if we can
                        if isinstance(raw_json_data, dict) and raw_json_data.get("api_name"):
                            json_api = raw_json_data.get("api_name").lower()
                            if json_api != api_name:
                                logger.info(f"JSON is for different API ({json_api}), not using it for {api_name}")
                                raw_json_data = None
                                continue
                        
                        logger.info(f"Using existing JSON: {json_path}")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load existing JSON ({json_path}): {e}")
                        raw_json_data = None

        # If no suitable JSON found or loading failed, or if forced, transcribe
        if not raw_json_data:
            if use_input_json:
                # This case should ideally not happen if load_json_data worked initially
                logger.error("Input was JSON but failed to load previously. Cannot transcribe.")
                return []

            logger.info(f"No suitable existing JSON found or --force used. Requesting new transcription for: {file_path}")
            
            # Initialize API instance
            api_instance = get_api_instance(api_name, api_key=kwargs.get('api_key'))
            if not api_instance:
                 logger.error(f"Could not initialize API: {api_name}. Skipping file.")
                 return []
            
            # Check if API key is valid before attempting transcription
            if not api_instance.check_api_key():
                logger.error(f"Invalid or missing API key for {api_name}. Please check your API key and try again.")
                return []
            
            # Get file size limit for the selected API
            size_limit_mb = get_api_file_size_limit(api_name)
            original_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            logger.info(f"Original file size: {original_size_mb:.2f}MB (limit: {size_limit_mb}MB)")
            
            # Proper processing sequence:
            # 1. Load the audio file
            # 2. Convert to FLAC (usually results in smaller file size)
            # 3. Check if FLAC file size is over limit
            # 4. If still too large, use chunking
            
            converted_file_path = None
            needs_chunking = False
            
            # Check if user wants to use the original input file
            if kwargs.get("use_input", False):
                logger.info("Using original input file as requested (--use-input)")
                # Check if original file exceeds the API limit
                if original_size_mb > size_limit_mb:
                    # If using original input is required but file is too large, abort
                    logger.error(f"Original file size ({original_size_mb:.2f}MB) exceeds the {size_limit_mb}MB limit for {api_name} API.")
                    logger.error(f"Cannot process this file with --use-input flag. Please remove the flag to enable conversion and chunking.")
                    return []
                # Use original file path
                file_path = original_file_path
            elif not file_ext.lower() in ['.flac']:
                # Convert to FLAC for most APIs
                logger.info("Converting to FLAC format for smaller file size")
                converted_file_path = convert_to_flac(file_path)
                if not converted_file_path:
                    logger.error("Failed to convert to FLAC. Skipping file.")
                    return []
                
                # Check new file size
                flac_size_mb = os.path.getsize(converted_file_path) / (1024 * 1024)
                logger.info(f"FLAC conversion complete. New size: {flac_size_mb:.2f}MB")
                
                if flac_size_mb > size_limit_mb:
                    logger.info(f"FLAC file size ({flac_size_mb:.2f}MB) exceeds {size_limit_mb}MB limit for {api_name} API.")
                    needs_chunking = True
                else:
                    # Use the FLAC file instead
                    file_path = converted_file_path
            else:
                # Already FLAC, just check size
                if original_size_mb > size_limit_mb:
                    logger.info(f"File size ({original_size_mb:.2f}MB) exceeds {size_limit_mb}MB limit for {api_name} API.")
                    needs_chunking = True
            
            # Prepare parameters for transcription
            transcribe_params = {}
            
            # Filter parameters to include only those relevant to the selected API
            # Common parameters for all APIs - Convert language code if provided
            if "language" in kwargs and kwargs["language"]:
                # Convert language code to API-specific format
                orig_language = kwargs["language"]
                api_language = get_language_code(orig_language, api_name)
                
                if api_language != orig_language:
                    logger.info(f"Converting language code '{orig_language}' to '{api_language}' for {api_name} API")
                
                transcribe_params["language"] = api_language
            else:
                transcribe_params["language"] = None
                
            # API-specific parameters
            if api_name == "assemblyai":
                # Language detection is true by default, only use specific language if provided
                if transcribe_params["language"]:
                    transcribe_params["language_detection"] = False
                else:
                    transcribe_params["language_detection"] = True
                
                # Speaker labels (diarization)
                transcribe_params["speaker_labels"] = kwargs.get("speaker_labels", False)
                
                # Always enable disfluencies (filler words like "um", "uh")
                transcribe_params["disfluencies"] = True
                
                # Model selection
                valid_assemblyai_models = ["best", "default", "nano", "small", "medium", "large", "auto"]
                model_value = kwargs.get("model")
                if not model_value or model_value not in valid_assemblyai_models:
                    logger.warning(f"Invalid AssemblyAI model: {model_value}, falling back to 'best'")
                    transcribe_params["model"] = "best"
                else:
                    transcribe_params["model"] = model_value
                
            elif api_name == "groq":
                valid_groq_models = ["whisper-large-v3", "whisper-medium", "whisper-small"]
                model_value = kwargs.get("model")
                # Only show warning if an invalid model is specified, not when None
                if model_value and model_value not in valid_groq_models:
                    logger.warning(f"Invalid Groq model: {model_value}, falling back to 'whisper-large-v3'")
                    transcribe_params["model"] = "whisper-large-v3"
                else:
                    # Set default or use valid specified model
                    transcribe_params["model"] = model_value if model_value else "whisper-large-v3"
            elif api_name == "openai":
                if "model" in kwargs:
                    transcribe_params["model"] = kwargs["model"]
                if "keep_flac" in kwargs:
                    transcribe_params["keep_flac"] = kwargs["keep_flac"]
                # Pass original file path for saving output to correct location
                transcribe_params["original_path"] = original_file_path
            elif api_name == "elevenlabs":
                if "model" in kwargs:
                    transcribe_params["model"] = kwargs["model"]

            # Start transcription process
            transcription_start_time = time.time()
            
            try:
                # If chunking is needed, use the appropriate method
                if needs_chunking:
                    chunk_length = kwargs.get("chunk_length", 600)  # Default 10 minutes
                    overlap = kwargs.get("overlap", 10)  # Default 10 seconds overlap
                    
                    logger.info(f"Using chunking with {chunk_length}s chunks and {overlap}s overlap")
                    
                    # Use the generic chunking function
                    api_result = transcribe_with_chunks(
                        file_path,
                        lambda chunk_path: api_instance.transcribe(chunk_path, **transcribe_params),
                        chunk_length=chunk_length,
                        overlap=overlap
                    )
                else:
                    # Call transcribe with filtered parameters
                    api_result = api_instance.transcribe(file_path, **transcribe_params)
                
                transcription_time = time.time() - transcription_start_time
                logger.info(f"API transcription completed in {transcription_time:.2f} seconds.")
                
                # Handle different result types from the API
                if hasattr(api_result, 'to_dict') and callable(getattr(api_result, 'to_dict')):
                    # This is already a TranscriptionResult object - use it directly
                    result_obj = api_result
                    basic_words = result_obj.words
                    raw_json_data = result_obj.to_dict()  # Convert to dict for saving
                    logger.debug(f"Received TranscriptionResult object from API")
                else:
                    # It's a raw dictionary - needs to be parsed
                    raw_json_data = api_result
                    if not raw_json_data:
                        raise Exception("API returned no data")
                    
                    # Check if we need to parse it
                    if isinstance(raw_json_data, dict) and raw_json_data.get("words"):
                        # It's already in a standard format with words
                        basic_words = raw_json_data.get("words", [])
                    else:
                        # We need to parse it based on API type
                        result_obj = parse_json_by_api(raw_json_data, api_name)
                        if not result_obj:
                            raise Exception(f"Failed to parse result from {api_name}")
                        
                        basic_words = result_obj.words
                        
                # Save raw JSON data if requested (always happens when using chunking)
                if save_cleaned_json or needs_chunking:
                    cleaned_json_path = file_dir / f"{file_name}-{api_name}-cleaned.json"
                    with open(cleaned_json_path, 'w', encoding='utf-8') as f:
                        # This is the standardized word format
                        json.dump({"api_name": api_name, "text": raw_json_data.get("text", ""), "words": basic_words}, f, indent=2, ensure_ascii=False)
                    logger.info(f"Saved cleaned standardized JSON to: {cleaned_json_path}")
                    output_files.append(str(cleaned_json_path))
            except Exception as e:
                # Check for rate limiting errors (429)
                if "429" in str(e) or "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                    retry_after = None
                    # Try to extract retry time from error message
                    error_msg = str(e)
                    
                    # Common patterns for retry information in error messages
                    retry_patterns = [
                        r"retry after (\d+)",
                        r"retry in (\d+)",
                        r"wait (\d+)",
                        r"available in (\d+)"
                    ]
                    
                    for pattern in retry_patterns:
                        match = re.search(pattern, error_msg, re.IGNORECASE)
                        if match:
                            retry_after = match.group(1)
                            break
                    
                    error_info = {
                        "error_type": "rate_limit_exceeded",
                        "api": api_name,
                        "file": str(file_path),
                        "message": error_msg,
                        "retry_after": retry_after
                    }
                    
                    # Clean up converted file if we created one
                    if converted_file_path and not kwargs.get("keep_flac", False):
                        try:
                            os.unlink(converted_file_path)
                            logger.info(f"Removed temporary converted file: {converted_file_path}")
                        except Exception as e2:
                            logger.warning(f"Failed to remove temporary file: {e2}")
                    
                    # Re-raise with structured error info for folder processing to handle
                    raise type(e)(f"RATE_LIMIT_ERROR: {json.dumps(error_info)}")
                else:
                    # Just raise the regular error
                    raise

        # At this point we have raw_json_data either from file or API
        # Create standardized TranscriptionResult object
        result = parse_json_by_api(raw_json_data, api_name)
        
        if not result:
            logger.error(f"Failed to parse result from {api_name}")
            return output_files
        
        # If we're reusing a loaded JSON, save a cleaned copy if requested
        if save_cleaned_json and use_input_json:
            cleaned_json_path = file_dir / f"{file_name}-{api_name}-cleaned.json"
            with open(cleaned_json_path, 'w', encoding='utf-8') as f:
                json.dump({"api_name": api_name, "text": result.text, "words": result.words}, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved cleaned standardized JSON to: {cleaned_json_path}")
            output_files.append(str(cleaned_json_path))
        
        # Generate outputs
        # Default to create standard SRT
        text_filepath = None
        srt_filepath = None
        
        # Always create text file
        text_filepath = file_dir / f"{file_name}.txt"
        create_text_file(result, text_filepath)
        logger.success(f"Created text transcript: {text_filepath}")
        output_files.append(str(text_filepath))
        
        # Create SRT based on parameters
        if kwargs.get("word_srt", False):
            srt_filepath = file_dir / f"{file_name}.srt"
            create_srt_file(result, srt_filepath, srt_mode="word")
            logger.success(f"Created word-level SRT: {srt_filepath}")
            output_files.append(str(srt_filepath))
        elif kwargs.get("davinci_srt", False):
            srt_filepath = file_dir / f"{file_name}.srt"
            create_srt_file(result, srt_filepath, srt_mode="davinci", show_pauses=kwargs.get("show_pauses", True))
            logger.success(f"Created DaVinci Resolve optimized SRT: {srt_filepath}")
            output_files.append(str(srt_filepath))
        else:
            # Standard SRT (default)
            srt_filepath = file_dir / f"{file_name}.srt"
            create_srt_file(result, srt_filepath, srt_mode="standard", fps=kwargs.get("fps", None))
            logger.success(f"Created standard SRT: {srt_filepath}")
            output_files.append(str(srt_filepath))
        
        # Clean up temp file if needed
        if 'converted_file_path' in locals() and converted_file_path and not kwargs.get("keep_flac", False) and converted_file_path != original_file_path:
            try:
                os.unlink(converted_file_path)
                logger.debug(f"Removed temporary converted file: {converted_file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file: {e}")
        
        # Total processing time
        total_time = time.time() - start_time
        logger.success(f"Completed processing in {total_time:.2f} seconds")
        
        return output_files
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        # Clean up converted file if we created one
        if 'converted_file_path' in locals() and converted_file_path and converted_file_path != original_file_path:
            try:
                os.unlink(converted_file_path)
                logger.debug(f"Removed temporary converted file: {converted_file_path}")
            except Exception:
                pass
        
        # Re-raise for folder processing to handle
        raise

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
        # Check if it's a JSON file for direct input
        if path.suffix.lower() == '.json' and kwargs.get("use_json_input", False):
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
            
            # If detected from filename, use that API, otherwise keep the one from parameters
            if api_name_from_file:
                logger.info(f"[PROCESSING] Detected API '{api_name_from_file}' from filename: {filename}")
                kwargs['api_name'] = api_name_from_file
            
            # Set use_input to True for JSON files
            kwargs['use_input'] = True
            
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
        
        # Regular audio file processing
        try:
            output_files = process_file(path, **kwargs)
            # Store the empty error tracker in kwargs
            kwargs['error_tracker'] = error_tracker
            return (1 if output_files else 0, 1)
        except Exception as e:
            error_message = str(e)
            
            # Rate limit error handling - same as above
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
    verbose: bool
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
        'use_json_input': use_json_input
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