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
from loguru import logger
from pydub import AudioSegment

# Import helpers
from transcribe_helpers.audio_processing import (
    check_audio_length, check_audio_format, convert_to_flac, convert_to_pcm
)
from transcribe_helpers.utils import setup_logger
from utils.parsers import TranscriptionResult, load_and_parse_json
from utils.transcription_api import get_api_instance
from utils.formatters import create_output_files


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


def check_json_exists(file_path: Union[str, Path], file_name: str) -> Tuple[bool, str]:
    """
    Check if a JSON transcript file exists for a given file name.
    
    Args:
        file_path: Directory containing the file
        file_name: Base name of the file without extension
        
    Returns:
        Tuple of (exists, json_path)
    """
    # First check for a generic JSON file (no API suffix)
    json_path = os.path.join(file_path, f"{file_name}.json")
    if os.path.exists(json_path):
        logger.info(f"Found generic JSON file: {json_path}")
        return True, json_path
    
    # Then check for API-specific JSON files
    for api_name in ["assemblyai", "elevenlabs", "groq", "openai"]:
        json_path = os.path.join(file_path, f"{file_name}_{api_name}.json")
        if os.path.exists(json_path):
            logger.info(f"Found {api_name.capitalize()} JSON file: {json_path}")
            return True, json_path
    
    return False, ""


def process_file(file_path: Union[str, Path], api_name: str, **kwargs) -> bool:
    """
    Process a single audio file with the specified API.
    
    Args:
        file_path: Path to the audio file
        api_name: Name of the API to use
        **kwargs: Additional parameters for processing
        
    Returns:
        True if processing was successful, False otherwise
    """
    try:
        file_path = Path(file_path)
        file_dir = file_path.parent
        file_name = file_path.stem
        
        # Always check for existing JSON first, regardless of other factors
        json_exists, json_path = check_json_exists(file_dir, file_name)
        if json_exists and not kwargs.get("force", False):
            logger.info(f"Using existing JSON file: {json_path}")
            
            # Parse the JSON file into our standardized format
            result = load_and_parse_json(json_path)
            
            # Create output files based on the existing JSON
            output_formats = kwargs.get("output_formats", ["text", "srt"])
            if kwargs.get("word_srt", False) and "word_srt" not in output_formats:
                output_formats.append("word_srt")
            if kwargs.get("davinci_srt", False) and "davinci_srt" not in output_formats:
                output_formats.append("davinci_srt")
            
            created_files = create_output_files(result, file_path, output_formats, **kwargs)
            logger.info(f"Created output files: {list(created_files.values())}")
            return True
        
        # Check if transcript already exists
        if check_transcript_exists(file_dir, file_name) and not kwargs.get("force", False):
            logger.info(f"Transcript for {file_name} already exists. Use --force to overwrite.")
            return False
        
        # Prepare the audio file
        use_input = kwargs.get("use_input", False)
        use_pcm = kwargs.get("use_pcm", False)
        
        processed_file = file_path
        if not use_input:
            # Convert to appropriate format
            if use_pcm:
                logger.info(f"Converting {file_path} to PCM WAV format")
                processed_file = convert_to_pcm(file_path)
            else:
                logger.info(f"Converting {file_path} to FLAC format")
                processed_file = convert_to_flac(file_path)
        
        # Get the API instance
        api_instance = get_api_instance(api_name)
        
        # Check API key
        if not api_instance.check_api_key():
            config_path = Path.home() / '.transcribe' / '.env'
            logger.error(f"Invalid or missing API key for {api_name}")
            logger.info(f"Please add your {api_name.upper()} API key to {config_path}")
            return False
        
        # Prepare API-specific parameters
        api_params = {}
        
        # Common parameters for all APIs
        api_params["language"] = kwargs.get("language")
        
        # API-specific parameters
        if api_name == "assemblyai":
            api_params["speaker_labels"] = kwargs.get("speaker_labels", True)
        elif api_name == "groq":
            api_params["model"] = kwargs.get("model", "whisper-large-v3")
            api_params["chunk_length"] = kwargs.get("chunk_length", 600)
            api_params["overlap"] = kwargs.get("overlap", 10)
        
        # Transcribe the audio
        result = api_instance.transcribe(processed_file, **api_params)
        
        # Clean up temporary files
        if not use_input and not kwargs.get("keep_flac", False) and processed_file != file_path:
            logger.info(f"Cleaning up temporary file: {processed_file}")
            try:
                os.remove(processed_file)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file: {e}")
        
        # Prepare output formats
        output_formats = kwargs.get("output_formats", ["text", "srt"])
        if kwargs.get("word_srt", False) and "word_srt" not in output_formats:
            output_formats.append("word_srt")
        if kwargs.get("davinci_srt", False) and "davinci_srt" not in output_formats:
            output_formats.append("davinci_srt")
        
        # Create output files
        created_files = create_output_files(result, file_path, output_formats, **kwargs)
        logger.info(f"Created output files: {list(created_files.values())}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        logger.exception(e)
        return False


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
                if file.lower().endswith(('.mp3', '.wav', '.ogg', '.mp4', '.flac', '.m4a', '.aac', '.wma')):
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
    debug: bool,
    verbose: bool
) -> None:
    """Transcribe audio files using various APIs with configurable output formats."""
    
    # --- .env management ---
    load_config_from_multiple_locations()
    new_user = ensure_user_config_directory()
    if new_user:
        logger.info(f"Welcome! Please edit your config at {Path.home() / '.transcribe' / '.env'} to add API keys.")
    # Setup logger
    setup_logger(debug, verbose)
    
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
    
    # Handle "all" output format
    output_formats = list(output)
    if "all" in output_formats:
        output_formats = ["text", "srt", "word_srt", "davinci_srt", "json"]
    
    # Process files
    start_time = time.time()
    logger.info(f"Starting transcription with {api.upper()} API")
    
    # Prepare parameters for process_audio_path
    params = {
        "language": language,
        "output_formats": output_formats,
        "chars_per_line": chars_per_line,
        "word_srt": word_srt,
        "davinci_srt": davinci_srt,
        "silentportions": silent_portions,
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
        "force": force
    }
    
    successful, total = process_audio_path(audio_path, api, **params)
    
    # Log results
    elapsed_time = time.time() - start_time
    logger.info(f"Transcription completed in {elapsed_time:.2f} seconds")
    logger.info(f"Successfully processed {successful}/{total} files")
    
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