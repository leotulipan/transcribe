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
from utils.formatters import create_output_files
from utils.parsers import TranscriptionResult, load_json_data, detect_and_parse_json
from utils.transcription_api import get_api_instance
from transcribe_helpers.text_processing import standardize_word_format, process_filler_words
from transcribe_helpers.chunking import split_audio, transcribe_with_chunks

# Assume helper functions like check_json_exists are defined later or imported
# from utils.file_utils import check_json_exists # Example

def check_json_exists(file_dir, file_name, api_name):
    # Placeholder implementation
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
    Process a single audio or JSON file.
    Handles transcription (if needed), parsing, standardization, cleaning, 
    and output file generation based on kwargs.
    
    Args:
        file_path: Path to the file to process
        **kwargs: Additional parameters, including api_name
    
    Returns:
        List of paths to the created output files.
    """
    file_path = Path(file_path)
    file_dir = file_path.parent
    file_name = file_path.stem
    
    # Get the API name from kwargs
    api_name = kwargs.get('api_name')
    if not api_name:
        logger.error(f"No API name provided for file: {file_path}")
        return []
    
    raw_json_data = None
    detected_api = api_name # Use provided API name initially
    processed_words = [] # This will hold our standardized intermediary format
    created_output_files = []
    
    # Store the original file path before any conversions
    original_file_path = file_path
    
    # 1. Get Raw JSON Data (either from existing file or new transcription)
    use_input_json = kwargs.get("use_input", False) and file_path.suffix.lower() == '.json'
    
    if use_input_json:
        logger.info(f"Using existing JSON file as input: {file_path}")
        raw_json_data = load_json_data(file_path)
        if not raw_json_data:
             logger.error(f"Failed to load input JSON: {file_path}. Skipping file.")
             return [] # Indicate failure for this file
    else:
        # Check if a pre-existing API-specific JSON exists
        json_exists, existing_json_path = check_json_exists(file_dir, file_name, api_name)
        
        if json_exists and not kwargs.get("force", False):
            logger.info(f"Using existing API JSON file: {existing_json_path}")
            raw_json_data = load_json_data(existing_json_path)
            if not raw_json_data:
                 logger.warning(f"Failed to load existing API JSON: {existing_json_path}. Will attempt transcription.")
            # Try to get api_name from the loaded json if not provided
            if not detected_api and raw_json_data and 'api_name' in raw_json_data:
                 detected_api = raw_json_data['api_name']
                 logger.info(f"Detected API '{detected_api}' from existing JSON.")

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
            
            # Skip conversion if --use-input is specified
            if not kwargs.get("use_input", False):
                logger.info(f"Converting input to FLAC format for optimal processing")
                
                if kwargs.get("use_pcm", False):
                    logger.info(f"Using PCM format as requested (larger file size)")
                    converted_file_path = convert_to_pcm(file_path)
                else:
                    converted_file_path = convert_to_flac(file_path)
                
                if not converted_file_path:
                    logger.error(f"Failed to convert audio file to proper format. Skipping file.")
                    return []
                
                file_path = converted_file_path
                logger.info(f"Conversion complete: {file_path}")
            
            # Now check file size AFTER conversion (FLAC is typically smaller)
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            logger.info(f"File size after conversion: {file_size_mb:.2f}MB (limit: {size_limit_mb}MB)")
            
            # Determine if chunking is needed
            if file_size_mb > size_limit_mb:
                if api_name in ["groq", "openai", "assemblyai"]:
                    logger.info(f"File size ({file_size_mb:.2f}MB) exceeds {size_limit_mb}MB limit for {api_name} API. Will use chunking.")
                    needs_chunking = True
                else:
                    logger.error(f"File size ({file_size_mb:.2f}MB) exceeds {size_limit_mb}MB limit for {api_name} API. Aborting.")
                    # Clean up converted file if we created one
                    if converted_file_path and not kwargs.get("keep_flac", False):
                        try:
                            os.unlink(converted_file_path)
                            logger.info(f"Removed temporary converted file: {converted_file_path}")
                        except Exception as e:
                            logger.warning(f"Failed to remove temporary file: {e}")
                    return []
                
            if file_size_mb > 0.9 * size_limit_mb:
                logger.warning(f"File size ({file_size_mb:.2f}MB) is close to the {size_limit_mb}MB limit for {api_name} API.")
                 
            transcription_start_time = time.time()
            try:
                 # Prepare transcription parameters
                 transcribe_params = {}
                 
                 # Add language parameter if provided
                 if "language" in kwargs and kwargs["language"]:
                     transcribe_params["language"] = kwargs["language"]
                     
                 # Add other API-specific parameters as needed
                 if api_name == "assemblyai":
                     if "speaker_labels" in kwargs:
                         transcribe_params["speaker_labels"] = kwargs["speaker_labels"]
                     if "dual_channel" in kwargs:
                         transcribe_params["dual_channel"] = kwargs["dual_channel"]
                     if "model" in kwargs:
                         transcribe_params["model"] = kwargs["model"]
                 elif api_name == "groq":
                     if "model" in kwargs:
                         transcribe_params["model"] = kwargs["model"]
                 elif api_name == "openai":
                     if "model" in kwargs:
                         transcribe_params["model"] = kwargs["model"]
                     if "keep_flac" in kwargs:
                         transcribe_params["keep_flac"] = kwargs["keep_flac"]
                     # Pass original file path for saving output to correct location
                     transcribe_params["original_path"] = original_file_path
                 
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
                     logger.debug(f"Received TranscriptionResult object from API")
                 else:
                     # It's a raw dictionary - needs to be parsed
                     raw_json_data = api_result
                     if not raw_json_data:
                         raise Exception("API returned no data")
                 
            except Exception as e:
                 logger.error(f"API transcription failed for {file_path}: {e}")
                 # Clean up converted file if we created one
                 if converted_file_path and not kwargs.get("keep_flac", False):
                     try:
                         os.unlink(converted_file_path)
                         logger.info(f"Removed temporary converted file after failed transcription: {converted_file_path}")
                     except Exception as e2:
                         logger.warning(f"Failed to remove temporary file: {e2}")
                 return []
            
            # Clean up temporary files
            if converted_file_path and not kwargs.get("keep_flac", False):
                try:
                    os.unlink(converted_file_path)
                    logger.info(f"Removed temporary converted file: {converted_file_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file: {e}")
    
    # --- At this point, we should have either raw_json_data or result_obj ---
    # If we have raw_json_data, we need to parse it
    if 'raw_json_data' in locals() and raw_json_data:
        try:
             # If API wasn't known before, try to detect it now from the raw data
             if not detected_api or detected_api == 'unknown':
                  detected_api, parse_result = detect_and_parse_json(raw_json_data)
                  logger.info(f"Detected API as '{detected_api}' during parsing.")
             else:
                  # Use the known API parser directly
                  _, parse_result = detect_and_parse_json(raw_json_data) # Reuse detection logic for now
                  
             if not parse_result:
                 logger.warning(f"Parsing resulted in empty word list for {file_path}. Skipping file.")
                 return []
                 
             # If parse_result is a TranscriptionResult, use it directly
             if hasattr(parse_result, 'words') and hasattr(parse_result, 'to_dict'):
                 result_obj = parse_result
                 basic_words = result_obj.words
             else:
                 # It's a list of words
                 basic_words = parse_result
                 
        except Exception as e:
            logger.error(f"Error parsing raw JSON data for {file_path}: {e}", exc_info=True)
            return []

    # 3. Standardize Format (Add Spacing, ensure ms format)
    try:
        standardize_silent_portions = kwargs.get("silent_portions", 0)
        standardize_show_pauses = kwargs.get("show_pauses", False) or standardize_silent_portions > 0
        
        # We should now have either result_obj (TranscriptionResult) or basic_words (list)
        if 'result_obj' in locals() and result_obj:
            # Already have a TranscriptionResult, use its words directly
            processed_words = result_obj.words
            logger.debug(f"Using pre-processed words from TranscriptionResult")
        else:
            # Process the basic_words list
            processed_words = standardize_word_format(
                basic_words,
                show_pauses=standardize_show_pauses,
                silence_threshold=standardize_silent_portions
            )
            logger.debug(f"Standardization complete. Word count: {len(processed_words)}")
    except Exception as e:
        logger.error(f"Error standardizing word format for {file_path}: {e}", exc_info=True)
        return []

    # 4. Save Cleaned JSON (Intermediary Format) if requested
    cleaned_json_path = None
    if kwargs.get("save_cleaned_json", False):
        # Determine output path for cleaned JSON
        # If input was a JSON, create cleaned path based on that
        if use_input_json:
            cleaned_json_path = file_dir / f"{file_name}_cleaned.json"
        else:
            # If input was audio, base it on audio name
             cleaned_json_path = file_dir / f"{file_path.stem}_cleaned.json"
             
        logger.info(f"Saving cleaned and standardized JSON to: {cleaned_json_path}")
        try:
            # Use a temporary TranscriptionResult object just for saving
            # This leverages the existing save method with timestamp formatting
            temp_result = TranscriptionResult(words=processed_words, api_name=detected_api)
            temp_result.save_words_only(cleaned_json_path)
            created_output_files.append(str(cleaned_json_path))
        except Exception as e:
            logger.error(f"Failed to save cleaned JSON to {cleaned_json_path}: {e}")
            # Continue processing even if saving cleaned JSON fails

    # 5. Apply Post-Processing (Fillers, etc.) to the standardized list
    words_for_output = processed_words # Start with the standardized list
    if kwargs.get("remove_fillers", False):
        try:
            filler_pause_threshold = kwargs.get("silent_portions", 250) # Use silent_portions as threshold
            logger.debug(f"Applying filler word removal. Current word count: {len(words_for_output)}")
            words_for_output = process_filler_words(
                words_for_output,
                pause_threshold=filler_pause_threshold,
                # filler_words=None will use defaults e.g. ["äh", "ähm"]
            )
            logger.debug(f"Word count after filler removal: {len(words_for_output)}")
        except Exception as e:
            logger.error(f"Error during filler word removal for {file_path}: {e}", exc_info=True)
            # Decide if we should continue or return []
            # Let's continue with the words we have so far

    # --- Add other post-processing steps here if needed --- 

    # 6. Generate Final Output Files (TXT, SRT, etc.)
    output_formats = kwargs.get("output", ["text", "srt"])
    # Use a temporary result object containing the final processed words for formatters
    final_result_obj = TranscriptionResult(words=words_for_output, api_name=detected_api)
    
    try:
         # Use the original audio path for output files, not the converted temp file
         output_base_path = original_file_path
         # Convert Click's tuple of output formats to a list
         output_format_list = list(output_formats)
         output_files = create_output_files(final_result_obj, output_base_path, output_format_list, **kwargs)
         logger.info(f"Created output files: {list(output_files.values())}")
         created_output_files.extend(list(output_files.values()))
    except Exception as e:
         logger.error(f"Error creating output files for {file_path}: {e}", exc_info=True)
         # Return empty list or partial list depending on desired behavior
         return [] 
         
    return created_output_files

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
    
    if not api_name:
        logger.error("No API name provided")
        return (0, 0)
    
    # Handle directory processing
    if path.is_dir():
        logger.info(f"Processing directory: {path}")
        
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
        for file in unique_files:
            logger.info(f"Processing file {successful+1}/{len(unique_files)}: {file}")
            output_files = process_file(file, **kwargs)
            if output_files:
                successful += 1
        
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
                logger.info(f"Detected API '{api_name_from_file}' from filename: {filename}")
                kwargs['api_name'] = api_name_from_file
            
            # Set use_input to True for JSON files
            kwargs['use_input'] = True
            
            output_files = process_file(path, **kwargs)
            return (1 if output_files else 0, 1)
        
        # Regular audio file processing
        output_files = process_file(path, **kwargs)
        return (1 if output_files else 0, 1)
    
    logger.error(f"Path not found or invalid: {path}")
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
    help="Model to use for transcription. API-specific options: groq=[*whisper-large-v3, whisper-medium, whisper-small], openai=[*whisper-1], assemblyai=[*best, default, nano, small, medium, large, auto]. Use nano for faster processing of short/simple audio.",
    default="whisper-large-v3"
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