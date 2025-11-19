"""
Unified Audio Transcription Tool - CLI Interface

This module provides a command-line interface for transcribing audio files
using different APIs with options for different output formats.
"""

import os
import sys
import json
import time
import glob
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import click
from loguru import logger
from dotenv import load_dotenv

# Import helpers from the package
from audio_transcribe.transcribe_helpers.audio_processing import (
    check_audio_length, check_audio_format, convert_to_flac, convert_to_pcm, 
    get_api_file_size_limit, check_file_size
)
from audio_transcribe.transcribe_helpers.utils import setup_logger
from audio_transcribe.utils.parsers import TranscriptionResult, load_json_data, detect_and_parse_json
# Import the whole parsers module to use getattr
from audio_transcribe.utils import parsers as all_parsers
from audio_transcribe.utils.formatters import create_output_files
from audio_transcribe.utils.api import get_api_instance

def check_json_exists(file_dir: Union[str, Path], file_name: str, api_name: str) -> Tuple[bool, Optional[Path]]:
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
    
    logger.debug(f"Processing file: {file_path}")
    logger.debug(f"API: {api_name}, Force: {force}, Save JSON: {save_cleaned_json}, Use JSON input: {use_json_input}")
    
    try:
        start_time = time.time()
        
        # Convert to Path object if it's a string
        original_input_path = Path(file_path) if isinstance(file_path, str) else file_path
        
        # Extract directory and file name components
        file_dir = original_input_path.parent
        file_name_stem = original_input_path.stem
        
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
                        match = re.search(r'_([a-zA-Z0-9]+)(?:_raw)?(?:_cleaned)?\.json$', original_input_path.name)
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
                        transcription_result = None 
                else:
                    logger.error(f"Failed to load data from JSON input file: {original_input_path}")
                    transcription_result = None 
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
        if transcription_result is None and not use_json_input:
            if not original_input_path.exists():
                logger.error(f"Audio file for transcription not found: {original_input_path}")
                return []
            current_audio_file_path = original_input_path 

            logger.info(f"Requesting new transcription for: {current_audio_file_path} using API: {api_name}")
            
            api_instance = get_api_instance(api_name, api_key=kwargs.get('api_key'))
            if not api_instance:
                 logger.error(f"Could not initialize API: {api_name}. Skipping file.")
                 return []
            if not api_instance.check_api_key():
                logger.error(f"Invalid or missing API key for {api_name}.")
                return []

            # File size checks
            if api_name.lower() != 'elevenlabs':
                max_size_mb = get_api_file_size_limit(api_name)
                current_size_mb = os.path.getsize(current_audio_file_path) / (1024 * 1024)
                logger.debug(f"File size: {current_size_mb:.2f}MB, API limit: {max_size_mb}MB")
                
                if current_size_mb > max_size_mb:
                    logger.error(f"File size ({current_size_mb:.2f}MB) exceeds {max_size_mb}MB limit for {api_name} API. Aborting")
                    return []

            # Pass all kwargs to the API instance's transcribe method
            transcribe_kwargs = kwargs.copy()
            transcribe_kwargs['original_path'] = current_audio_file_path
            
            # Handle model parameters
            if api_name == "elevenlabs":
                transcribe_kwargs['model_id'] = kwargs.get('model', 'scribe_v1')
            else:
                if api_name == "groq":
                    transcribe_kwargs['model'] = kwargs.get('model', 'whisper-large-v3')
                elif api_name == "openai":
                    transcribe_kwargs['model'] = kwargs.get('model', 'whisper-1')
                elif api_name == "assemblyai":
                    transcribe_kwargs['model'] = kwargs.get('model', 'best')
                else:
                    if 'model' in kwargs and kwargs['model'] is not None:
                        transcribe_kwargs['model'] = kwargs['model']
            
            # Perform transcription
            api_call_response = api_instance.transcribe(current_audio_file_path, **transcribe_kwargs)

            # Handle API response
            if isinstance(api_call_response, TranscriptionResult):
                transcription_result = api_call_response
            elif isinstance(api_call_response, dict):
                # Fallback for raw dicts (though APIs should return TranscriptionResult now)
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

        if not transcription_result or not isinstance(transcription_result, TranscriptionResult):
            logger.error(f"Failed to obtain a valid transcription result for {original_input_path}.")
            return []

        logger.success(f"Successfully obtained transcription. Words count: {len(transcription_result.words)}")
        
        # Output generation
        output_format_types = kwargs.get("output", ["text", "srt"])
        if "all" in output_format_types:
            output_format_types = ["text", "srt", "word_srt", "davinci_srt", "json"]
        
        created_files_dict = create_output_files(
            result=transcription_result,
            audio_path=original_input_path,
            format_types=output_format_types,
            **kwargs
        )
        output_files = list(created_files_dict.values())
        
        if save_cleaned_json:
            cleaned_json_path = file_dir / f"{file_name_stem}_{api_name}_cleaned.json"
            transcription_result.save(cleaned_json_path)
            logger.info(f"Saved cleaned (parsed) JSON to: {cleaned_json_path}")
            output_files.append(str(cleaned_json_path))

        total_time = time.time() - start_time
        logger.success(f"Completed processing {original_input_path} in {total_time:.2f} seconds. Output files: {output_files}")
        return output_files
        
    except Exception as e:
        logger.error(f"Unhandled error processing file {file_path}: {str(e)}")
        if debug:
            import traceback
            logger.debug(traceback.format_exc())
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
    
    # Handle directory processing
    if path.is_dir():
        logger.info(f"[PROCESSING] Directory: {path}")
        
        audio_extensions = ['.mp3', '.wav', '.flac', '.m4a', '.ogg', '.opus']
        video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.webm']
        all_media_extensions = audio_extensions + video_extensions
        
        media_files = []
        for ext in all_media_extensions:
            media_files.extend(list(path.glob(f"*{ext}")))
        
        if not media_files:
            for ext in all_media_extensions:
                media_files.extend(list(path.glob(f"*{ext.upper()}")))
        
        if not media_files:
            logger.error(f"No audio or video files found in directory: {path}")
            return (0, 0)
        
        # Group files by basename
        basename_groups = {}
        for file in media_files:
            basename = file.stem
            if basename in basename_groups:
                basename_groups[basename].append(file)
            else:
                basename_groups[basename] = [file]
        
        unique_files = []
        for basename, files in basename_groups.items():
            if len(files) == 1:
                unique_files.append(files[0])
            else:
                # Prioritize video > flac > wav > others
                video_files = [f for f in files if f.suffix.lower() in video_extensions]
                if video_files:
                    unique_files.append(max(video_files, key=lambda f: f.stat().st_size))
                else:
                    flac_files = [f for f in files if f.suffix.lower() == '.flac']
                    if flac_files:
                        unique_files.append(flac_files[0])
                    else:
                        wav_files = [f for f in files if f.suffix.lower() == '.wav']
                        if wav_files:
                            unique_files.append(wav_files[0])
                        else:
                            unique_files.append(max(files, key=lambda f: f.stat().st_size))
        
        logger.info(f"Found {len(unique_files)} unique audio/video files to process")
        
        successful = 0
        for i, file in enumerate(unique_files):
            logger.info(f"[PROCESSING] File {i+1}/{len(unique_files)}: {file}")
            if process_file(file, **kwargs):
                successful += 1
        
        return (successful, len(unique_files))
    
    # Process single file
    if path.is_file():
        # Auto-apply --use-json-input if file is a JSON file
        if path.suffix.lower() == '.json':
            logger.info(f"Auto-enabling JSON input mode for file: {path}")
            kwargs['use_json_input'] = True
            
            if api_name == "auto" or not api_name:
                filename = path.stem
                if "_assemblyai" in filename: kwargs['api_name'] = "assemblyai"
                elif "_elevenlabs" in filename: kwargs['api_name'] = "elevenlabs"
                elif "_groq" in filename: kwargs['api_name'] = "groq"
                elif "_openai" in filename: kwargs['api_name'] = "openai"
        
        if process_file(path, **kwargs):
            return (1, 1)
        return (0, 1)
    
    logger.error(f"Path not found or invalid: {path}")
    return (0, 0)

@click.group(invoke_without_command=True)
@click.option("--file", "-f", type=click.Path(exists=True), help="Audio/video file to transcribe")
@click.option("--folder", "-F", type=click.Path(exists=True, file_okay=False, dir_okay=True), help="Folder containing audio/video files to transcribe")
@click.option("--api", "-a", type=click.Choice(["assemblyai", "elevenlabs", "groq", "openai"], case_sensitive=False), default="groq", help="API to use for transcription (default: groq)")
@click.option("--language", "-l", help="Language code (ISO-639-1 or ISO-639-3)", default=None)
@click.option("--output", "-o", type=click.Choice(["text", "srt", "word_srt", "davinci_srt", "json", "all"], case_sensitive=False), multiple=True, default=["text", "srt"], help="Output format(s) to generate (default: text,srt)")
@click.option("--chars-per-line", "-c", type=int, default=80, help="Maximum characters per line in SRT file (default: 80)")
@click.option("--words-per-subtitle", "-w", type=int, default=0, help="Maximum words per subtitle block (default: 0 = disabled). Mutually exclusive with -c.")
@click.option("--word-srt", "-C", is_flag=True, help="Output SRT with each word as its own subtitle")
@click.option("--davinci-srt", "-D", is_flag=True, help="Output SRT optimized for DaVinci Resolve")
@click.option("--silent-portions", "-p", type=int, default=0, help="Mark pauses longer than X milliseconds with (...)")
@click.option("--padding-start", type=int, default=0, help="Milliseconds to offset word start times (negative=earlier, positive=later)")
@click.option("--padding-end", type=int, default=0, help="Milliseconds to offset word end times (negative=earlier, positive=later)")
@click.option("--show-pauses", is_flag=True, help="Add (...) text for pauses longer than silent-portions value")
@click.option("--filler-lines", is_flag=True, help="Output filler words as their own subtitle lines (uppercased)")
@click.option("--filler-words", multiple=True, help="Custom filler words to detect")
@click.option("--remove-fillers/--no-remove-fillers", default=False, help="Remove filler words like 'äh' and 'ähm' and treat them as pauses")
@click.option("--speaker-labels/--no-speaker-labels", default=True, help="Enable/disable speaker diarization (AssemblyAI only)")
@click.option("--fps", type=float, help="Frames per second for frame-based editing")
@click.option("--fps-offset-start", type=int, default=-1, help="Frames to offset from start time")
@click.option("--fps-offset-end", type=int, default=0, help="Frames to offset from end time")
@click.option("--diarize/--no-diarize", default=False, help="Enable speaker diarization")
@click.option("--num-speakers", type=int, default=None, help="Maximum number of speakers (1..32). Requires --diarize.")
@click.option("--use-input", is_flag=True, help="Use original input file without conversion")
@click.option("--use-pcm", is_flag=True, help="Convert to PCM WAV format instead of FLAC")
@click.option("--keep-flac", is_flag=True, help="Keep the generated FLAC file after processing")
@click.option("--model", "-m", help="Model to use for transcription", default=None)
@click.option("--chunk-length", type=int, default=600, help="Length of each chunk in seconds for long audio")
@click.option("--overlap", type=int, default=10, help="Overlap between chunks in seconds")
@click.option("--force", "-r", is_flag=True, help="Force re-transcription even if transcript exists")
@click.option("--save-cleaned-json", "-J", is_flag=True, help="Save the cleaned and consistent pre-processed JSON file")
@click.option("--use-json-input", "-j", is_flag=True, help="Accept JSON files as input")
@click.option("--debug", "-d", is_flag=True, help="Enable debug logging")
@click.option("--verbose", "-v", is_flag=True, help="Show all log messages in console")
@click.option("--start-hour", type=int, default=None, help="Hour offset for SRT timestamps")
@click.pass_context
def main(ctx, file, folder, api, language, output, chars_per_line, words_per_subtitle, 
         word_srt, davinci_srt, silent_portions, padding_start, padding_end, show_pauses, 
         filler_lines, filler_words, remove_fillers, speaker_labels, fps, fps_offset_start, 
         fps_offset_end, diarize, num_speakers, use_input, use_pcm, keep_flac, model, 
         chunk_length, overlap, force, save_cleaned_json, use_json_input, debug, verbose, start_hour):
    """Unified Audio Transcription Tool."""
    
    # If a subcommand is invoked, do nothing here (let the subcommand handle it)
    if ctx.invoked_subcommand is not None:
        return

    # Validate input parameters for default action
    if not file and not folder:
        click.echo(ctx.get_help())
        sys.exit(0)
    
    if file and folder:
        click.echo("Error: Cannot specify both --file and --folder. Choose one option.")
        sys.exit(1)
    
    # Set up logging
    setup_logger(debug=debug, verbose=verbose)
    
    # Load environment variables
    load_dotenv()
    
    # Process DaVinci mode defaults
    if davinci_srt:
        if chars_per_line == 80: chars_per_line = 500
        if silent_portions == 0: silent_portions = 250
        if padding_start == 0: padding_start = -125
        if not remove_fillers: remove_fillers = True
        if not show_pauses: show_pauses = True
    
    # Handle "all" output format
    output_formats = list(output)
    if "all" in output_formats:
        output_formats = ["text", "srt", "word_srt", "davinci_srt", "json"]
    
    # Prepare kwargs
    kwargs = {
        "api_name": api,
        "language": language,
        "output": output_formats,
        "chars_per_line": chars_per_line,
        "words_per_subtitle": words_per_subtitle,
        "word_srt": word_srt,
        "davinci_srt": davinci_srt,
        "silent_portions": silent_portions,
        "padding_start": padding_start,
        "padding_end": padding_end,
        "show_pauses": show_pauses,
        "filler_lines": filler_lines,
        "filler_words": filler_words,
        "remove_fillers": remove_fillers,
        "speaker_labels": speaker_labels,
        "fps": fps,
        "fps_offset_start": fps_offset_start,
        "fps_offset_end": fps_offset_end,
        "diarize": diarize,
        "num_speakers": num_speakers,
        "use_input": use_input,
        "use_pcm": use_pcm,
        "keep_flac": keep_flac,
        "model": model,
        "chunk_length": chunk_length,
        "overlap": overlap,
        "force": force,
        "save_cleaned_json": save_cleaned_json,
        "use_json_input": use_json_input,
        "debug": debug,
        "verbose": verbose,
        "start_hour": start_hour
    }
    
    # Run processing
    if folder:
        process_audio_path(folder, **kwargs)
    else:
        process_audio_path(file, **kwargs)

@main.command()
def setup():
    """Setup API keys and configuration."""
    click.echo("Setup wizard not implemented yet.")

@main.group()
def tools():
    """Auxiliary tools."""
    pass

@tools.command()
def join_srt():
    """Join SRT files."""
    click.echo("Join SRT tool not implemented yet.")

if __name__ == "__main__":
    main()