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
    check_audio_length, check_audio_format, convert_to_flac, convert_to_pcm
)
from transcribe_helpers.utils import setup_logger
from utils.formatters import create_output_files
from utils.parsers import TranscriptionResult, load_json_data, detect_and_parse_json
from utils.transcription_api import get_api_instance
from transcribe_helpers.text_processing import standardize_word_format, process_filler_words

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


def process_file(file_path: Union[str, Path], api_name: str, **kwargs) -> List[str]:
    """
    Process a single audio or JSON file.
    Handles transcription (if needed), parsing, standardization, cleaning, 
    and output file generation based on kwargs.
    
    Returns:
        List of paths to the created output files.
    """
    file_path = Path(file_path)
    file_dir = file_path.parent
    file_name = file_path.stem
    
    raw_json_data = None
    detected_api = api_name # Use provided API name initially
    processed_words = [] # This will hold our standardized intermediary format
    created_output_files = []
    
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
            api_instance = get_api_instance(api_name, api_key=kwargs.get('api_key'))
            if not api_instance:
                 logger.error(f"Could not initialize API: {api_name}. Skipping file.")
                 return []
                 
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
                 elif api_name == "groq":
                     if "model" in kwargs:
                         transcribe_params["model"] = kwargs["model"]
                     if "chunk_length" in kwargs:
                         transcribe_params["chunk_length"] = kwargs["chunk_length"]
                     if "overlap" in kwargs:
                         transcribe_params["overlap"] = kwargs["overlap"]
                 elif api_name == "openai":
                     if "model" in kwargs:
                         transcribe_params["model"] = kwargs["model"]
                 
                 # Call transcribe with filtered parameters
                 raw_json_data = api_instance.transcribe(file_path, **transcribe_params)
                 transcription_time = time.time() - transcription_start_time
                 logger.info(f"API transcription completed in {transcription_time:.2f} seconds.")
                 
                 if not raw_json_data:
                      raise Exception("API returned no data")
                      
                 # Save raw API output if desired (optional)
                 # raw_output_path = file_dir / f"{file_name}_{api_name}_raw.json"
                 # with open(raw_output_path, 'w', encoding='utf-8') as f:
                 #    json.dump(raw_json_data, f, indent=2, ensure_ascii=False)
                 # logger.info(f"Saved raw API output to {raw_output_path}")
                 
            except Exception as e:
                 logger.error(f"API transcription failed for {file_path}: {e}")
                 return [] # Indicate failure

    # --- At this point, raw_json_data should contain the data --- 
    if not raw_json_data:
         logger.error(f"Failed to obtain transcription data for {file_path}. Skipping.")
         return []

    # 2. Parse Raw Data into Basic Word List (structure + int ms times)
    try:
         # If API wasn't known before, try to detect it now from the raw data
         if not detected_api or detected_api == 'unknown':
              detected_api, basic_words = detect_and_parse_json(raw_json_data)
              logger.info(f"Detected API as '{detected_api}' during parsing.")
         else:
              # Use the known API parser directly (implement this logic in detect_and_parse_json or here)
              _, basic_words = detect_and_parse_json(raw_json_data) # Reuse detection logic for now
              # Alternatively, have specific calls: 
              # if detected_api == 'assemblyai': basic_words = parse_assemblyai_format(raw_json_data)
              # etc.
              
         if not basic_words:
             logger.warning(f"Parsing resulted in empty word list for {file_path}. Skipping file.")
             return []
             
    except Exception as e:
         logger.error(f"Error parsing raw JSON data for {file_path}: {e}", exc_info=True)
         return []

    # 3. Standardize Format (Add Spacing, ensure ms format)
    try:
        standardize_silent_portions = kwargs.get("silent_portions", 0)
        standardize_show_pauses = kwargs.get("show_pauses", False) or standardize_silent_portions > 0
        
        # Check if we received a TranscriptionResult object or a raw word list
        if hasattr(basic_words, 'words'):
            # We have a TranscriptionResult object
            result_obj = basic_words
            basic_words_list = result_obj.words
            logger.debug(f"Received TranscriptionResult object with {len(basic_words_list)} words")
        else:
            # We have a raw word list
            basic_words_list = basic_words
            logger.debug(f"Received raw word list with {len(basic_words_list)} words")
        
        processed_words = standardize_word_format(
            basic_words_list,
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
            temp_result = TranscriptionResult(words=processed_words)
            temp_result.save_words_only(cleaned_json_path)
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
    output_formats = kwargs.get("output_formats", ["text", "srt"])
    # Use a temporary result object containing the final processed words for formatters
    final_result_obj = TranscriptionResult(words=words_for_output, api_name=detected_api)
    
    # Add explicit print debugging for first few words
    print("DEBUG: First 5 words for SRT generation:")
    for i, word in enumerate(words_for_output[:5]):
        print(f"Word {i}: {word}")
    if "spacing" in [w.get("type", "") for w in words_for_output[:10]]:
        print("DEBUG: Spacing detected in first 10 words")
    else:
        print("DEBUG: No spacing detected in first 10 words")
    
    # Debug the silentportions parameter
    silentportions = kwargs.get("silent_portions", 0)
    print(f"DEBUG: silent_portions parameter value = {silentportions}")
        
    try:
         # Use the original audio path (or input JSON path if used) to base output names
         output_base_path = file_path if use_input_json else file_path 
         created_output_files = create_output_files(final_result_obj, output_base_path, output_formats, **kwargs)
         logger.info(f"Created output files: {list(created_output_files.values())}")
    except Exception as e:
         logger.error(f"Error creating output files for {file_path}: {e}", exc_info=True)
         # Return empty list or partial list depending on desired behavior
         return [] 
         
    return list(created_output_files.values())


def process_audio_path(audio_path: str, api_name: str, **kwargs) -> Tuple[int, int]:
    """
    Process audio files based on the provided path (file, directory, or wildcard).
    
    Args:
        audio_path: Path to process (can be a file, directory, or wildcard pattern)
        api_name: Name of the API to use
        **kwargs: Additional parameters for processing
        
    Returns:
        Tuple of (successful_count, total_count)
    """
    # Initialize counters
    successful = 0
    total = 0
    
    # Dictionary to store file paths
    files_dict = {}
    
    # Check if audio_path is a directory
    if os.path.isdir(audio_path):
        logger.info(f"Processing directory: {audio_path}")
        for root, dirs, files in os.walk(audio_path):
            for file in files:
                # Include JSON files if use_json_input is True
                if kwargs.get("use_json_input", False) and file.lower().endswith(".json"):
                    files_dict[file] = os.path.join(root, file)
                elif file.lower().endswith(('.mp3', '.wav', '.ogg', '.mp4', '.flac', '.m4a', '.aac', '.wma')):
                    files_dict[file] = os.path.join(root, file)
    
    # Check if audio_path is a file
    elif os.path.isfile(audio_path):
        file_name = os.path.basename(audio_path)
        files_dict[file_name] = audio_path
        logger.info(f"Processing file: {audio_path}")
    
    # Check if audio_path is a wildcard pattern
    elif '*' in audio_path or '?' in audio_path:
        logger.info(f"Processing wildcard pattern: {audio_path}")
        for file_path in glob.glob(audio_path):
            if os.path.isfile(file_path):
                file_name = os.path.basename(file_path)
                # Include JSON files if use_json_input is True
                if kwargs.get("use_json_input", False) and file_path.lower().endswith(".json"):
                    files_dict[file_name] = file_path
                elif file_path.lower().endswith(('.mp3', '.wav', '.ogg', '.mp4', '.flac', '.m4a', '.aac', '.wma')):
                    files_dict[file_name] = file_path
    
    else:
        logger.error(f"Invalid audio path: {audio_path}")
        return 0, 0
    
    # Process each file
    total = len(files_dict)
    logger.info(f"Found {total} file(s) to process")
    
    for i, (file_name, file_path) in enumerate(files_dict.items(), 1):
        logger.info(f"Processing file {i}/{total}: {file_name}")
        if process_file(file_path, api_name, **kwargs):
            successful += 1
    
    return successful, total


@click.command()
@click.argument("audio_path", type=click.Path(exists=True))
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
    help="Model to use for transcription (Groq only)",
    default="whisper-large-v3"
)
@click.option(
    "--chunk-length",
    type=int,
    default=600,
    help="Length of each chunk in seconds for long audio (Groq only)"
)
@click.option(
    "--overlap",
    type=int,
    default=10,
    help="Overlap between chunks in seconds (Groq only)"
)
@click.option(
    "--force", "-f",
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
    audio_path: str,
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
    """
    Transcribe audio files using various APIs with options for different output formats.
    
    AUDIO_PATH can be a file, directory, or wildcard pattern.
    """
    # Set up logging
    setup_logger(debug=debug, verbose=verbose)
    
    # Load configuration
    load_config_from_multiple_locations()
    new_user = ensure_user_config_directory()
    if new_user:
        logger.info(f"Welcome! Please edit your config at {Path.home() / '.transcribe' / '.env'} to add API keys.")
    
    # Process DaVinci mode defaults
    if davinci_srt:
        if chars_per_line == 80:  # Only set if user didn't specify
            chars_per_line = 500
        if silent_portions == 0:   # Only set if user didn't specify
            silent_portions = 250
        if padding_start == 0:    # Only set if user didn't specify
            padding_start = -125
        if not remove_fillers:     # Only set if user didn't explicitly disable
            remove_fillers = True
        if not show_pauses:       # Only set if user didn't explicitly specify
            show_pauses = True
    
    # Process output formats
    output_formats = list(output)
    if "all" in output_formats:
        output_formats = ["text", "srt", "word_srt", "davinci_srt", "json"]
    
    # Start time tracking
    start_time = time.time()
    logger.info(f"Starting transcription with {api.upper()} API")
    
    # Process the audio file(s)
    kwargs = {
        "language": language,
        "output_formats": output_formats,
        "chars_per_line": chars_per_line,
        "word_srt": word_srt,
        "davinci_srt": davinci_srt,
        "silent_portions": silent_portions,
        "padding_start": padding_start,
        "padding_end": padding_end,
        "show_pauses": show_pauses,
        "remove_fillers": remove_fillers,
        "speaker_labels": speaker_labels,
        "fps": fps,
        "fps_offset_start": fps_offset_start,
        "fps_offset_end": fps_offset_end,
        "use_input": use_input,
        "use_pcm": use_pcm,
        "keep_flac": keep_flac,
        "model": model,
        "chunk_length": chunk_length,
        "overlap": overlap,
        "force": force,
        "save_cleaned_json": save_cleaned_json,
        "use_json_input": use_json_input,
    }
    
    successful, total = process_audio_path(audio_path, api, **kwargs)
    
    # Log results
    elapsed_time = time.time() - start_time
    logger.info(f"Transcription completed in {elapsed_time:.2f} seconds")
    logger.info(f"Processed {successful}/{total} files successfully")
    
    if successful != total:
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