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
import warnings

# Suppress pydub SyntaxWarnings (invalid escape sequence)
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pydub")

# Import helpers from the package
from audio_transcribe.transcribe_helpers.audio_processing import (
    check_audio_length, check_audio_format, convert_to_flac, convert_to_pcm, 
    get_api_file_size_limit, check_file_size, optimize_audio_for_api, OptimizationResult
)
from audio_transcribe.utils.api.chunking import ChunkingMixin
from audio_transcribe.transcribe_helpers.utils import setup_logger
from audio_transcribe.utils.parsers import TranscriptionResult, load_json_data, detect_and_parse_json
# Import the whole parsers module to use getattr
from audio_transcribe.utils import parsers as all_parsers
from audio_transcribe.utils.formatters import create_output_files
from audio_transcribe.utils.api import get_api_instance
from audio_transcribe.utils.config import ConfigManager
from audio_transcribe.utils.defaults import DefaultsManager

# Import TUI modules
from audio_transcribe.tui import run_setup_wizard, run_interactive_mode

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
    keep_optimized = kwargs.get("keep", False)
    
    logger.debug(f"Processing file: {file_path}")
    logger.debug(f"API: {api_name}, Force: {force}, Save JSON: {save_cleaned_json}, Use JSON input: {use_json_input}")
    
    # Track temporary files for cleanup
    temp_files_to_cleanup = []
    
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

            # Optimize audio file size if needed
            try:
                opt_result = optimize_audio_for_api(current_audio_file_path, api_name)
                max_size_mb = get_api_file_size_limit(api_name)
                
                # Check if file still exceeds limit and API supports chunking
                if not opt_result.fits_limit(max_size_mb) and isinstance(api_instance, ChunkingMixin):
                    # Calculate minimum chunk length to keep chunks under limit
                    if opt_result.bytes_per_second > 0:
                        # Calculate chunk length: max_size_bytes / bytes_per_second
                        max_size_bytes = max_size_mb * 1024 * 1024
                        calculated_chunk_length = max_size_bytes / opt_result.bytes_per_second
                        
                        # Use the smaller of calculated or user-specified chunk_length
                        user_chunk_length = kwargs.get("chunk_length", 600)
                        chunk_length_override = min(calculated_chunk_length, user_chunk_length)
                        
                        # Ensure minimum chunk length (at least 10 seconds)
                        chunk_length_override = max(chunk_length_override, 10)
                        
                        logger.info(f"File ({opt_result.size_mb:.2f}MB) exceeds {max_size_mb}MB limit. "
                                  f"Using chunking with length={chunk_length_override:.1f}s "
                                  f"(calculated from {opt_result.bytes_per_second:.0f} bytes/sec)")
                        
                        # Override chunk_length in kwargs
                        transcribe_kwargs['chunk_length'] = int(chunk_length_override)
                    else:
                        logger.warning("Could not calculate bytes per second. Using default chunk length.")
                elif not opt_result.fits_limit(max_size_mb):
                    logger.error(f"Audio optimization failed: File ({opt_result.size_mb:.2f}MB) exceeds "
                               f"{max_size_mb}MB limit and API does not support chunking.")
                    return []
                
                current_audio_file_path = opt_result.path
                if opt_result.is_temporary:
                    temp_files_to_cleanup.append(current_audio_file_path)
            except Exception as e:
                logger.error(f"Audio optimization failed: {e}")
                return []

            # Pass all kwargs to the API instance's transcribe method
            transcribe_kwargs = kwargs.copy()
            transcribe_kwargs['original_path'] = original_input_path # Pass original path for metadata
            
            # Handle model parameters
            # DefaultsManager should have already populated 'model' with default if needed
            if 'model' in kwargs and kwargs['model']:
                if api_name == "elevenlabs":
                    transcribe_kwargs['model_id'] = kwargs['model']
                else:
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
        output_files = create_output_files(transcription_result, original_input_path, kwargs)
        
        # Save cleaned JSON if requested
        if save_cleaned_json:
            cleaned_json_path = file_dir / f"{file_name_stem}_{api_name}_cleaned.json"
            try:
                with open(cleaned_json_path, 'w', encoding='utf-8') as f:
                    json.dump(transcription_result.to_dict(), f, indent=2, ensure_ascii=False)
                logger.info(f"Saved cleaned JSON to: {cleaned_json_path}")
                output_files.append(str(cleaned_json_path))
            except Exception as e:
                logger.error(f"Failed to save cleaned JSON: {e}")

        elapsed_time = time.time() - start_time
        logger.success(f"Processing completed in {elapsed_time:.2f}s")
        return output_files

    except Exception as e:
        logger.exception(f"An error occurred while processing {file_path}: {e}")
        return []
    finally:
        # Cleanup temporary files
        if not keep_optimized:
            for temp_file in temp_files_to_cleanup:
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                        logger.debug(f"Cleaned up temporary file: {temp_file}")
                    except Exception as e:
                        logger.warning(f"Failed to delete temporary file {temp_file}: {e}")

def process_audio_path(path: Union[str, Path], **kwargs) -> None:
    """
    Process a file or directory of audio files.
    """
    path_obj = Path(path)
    if path_obj.is_file():
        process_file(path_obj, **kwargs)
    elif path_obj.is_dir():
        # Recursive search for audio/video files
        extensions = ['.mp3', '.wav', '.m4a', '.mp4', '.mkv', '.flac', '.ogg', '.webm']
        files = []
        for ext in extensions:
            files.extend(path_obj.rglob(f"*{ext}"))
        
        if not files:
            logger.warning(f"No audio/video files found in {path}")
            return
            
        logger.info(f"Found {len(files)} files to process in {path}")
        for file in files:
            process_file(file, **kwargs)
    else:
        logger.error(f"Path not found: {path}")

@click.group(invoke_without_command=True)
@click.argument("input_path", required=False, type=click.Path(exists=True))
@click.option("--file", "-f", help="Input audio/video file (legacy)")
@click.option("--folder", "-F", help="Input folder containing audio/video files (legacy)")
@click.option("--api", "-a", help="API to use (groq, openai, assemblyai, elevenlabs)")
@click.option("--language", "-l", help="Language code (e.g., en, de, fr)")
@click.option("--output", "-o", type=click.Choice(["text", "srt", "word_srt", "davinci_srt", "json", "all"], case_sensitive=False), multiple=True, help="Output format(s) to generate (default: text,srt)")
@click.option("--chars-per-line", "-c", type=int, default=None, help="Maximum characters per line in SRT file (default: 80)")
@click.option("--words-per-subtitle", "-w", type=int, default=None, help="Maximum words per subtitle block (default: 0 = disabled). Mutually exclusive with -c.")
@click.option("--word-srt", "-C", is_flag=True, help="Output SRT with each word as its own subtitle")
@click.option("--davinci-srt", "-D", is_flag=True, help="Output SRT optimized for DaVinci Resolve")
@click.option("--silent-portions", "-p", type=int, default=None, help="Mark pauses longer than X milliseconds with (...)")
@click.option("--padding-start", type=int, default=None, help="Milliseconds to offset word start times (negative=earlier, positive=later)")
@click.option("--padding-end", type=int, default=None, help="Milliseconds to offset word end times (negative=earlier, positive=later)")
@click.option("--show-pauses", is_flag=True, default=None, help="Add (...) text for pauses longer than silent-portions value")
@click.option("--filler-lines", is_flag=True, help="Output filler words as their own subtitle lines (uppercased)")
@click.option("--filler-words", multiple=True, help="Custom filler words to detect")
@click.option("--remove-fillers/--no-remove-fillers", default=None, help="Remove filler words like 'äh' and 'ähm' and treat them as pauses")
@click.option("--speaker-labels/--no-speaker-labels", default=None, help="Enable/disable speaker diarization (AssemblyAI only)")
@click.option("--fps", type=float, help="Frames per second for frame-based editing")
@click.option("--fps-offset-start", type=int, default=-1, help="Frames to offset from start time")
@click.option("--fps-offset-end", type=int, default=0, help="Frames to offset from end time")
@click.option("--diarize/--no-diarize", default=False, help="Enable speaker diarization")
@click.option("--num-speakers", type=int, default=None, help="Maximum number of speakers (1..32). Requires --diarize.")
@click.option("--use-input", is_flag=True, help="Use original input file without conversion")
@click.option("--use-pcm", is_flag=True, help="Convert to PCM WAV format instead of FLAC")
@click.option("--keep-flac", is_flag=True, help="Keep the generated FLAC file after processing")
@click.option("--keep", is_flag=True, help="Keep optimized/converted audio files")
@click.option("--model", "-m", help="Model to use for transcription", default=None)
@click.option("--chunk-length", type=int, default=600, help="Length of each chunk in seconds for long audio")
@click.option("--overlap", type=int, default=10, help="Overlap between chunks in seconds")
@click.option("--force", "-r", is_flag=True, help="Force re-transcription even if transcript exists")
@click.option("--save-cleaned-json", "-J", is_flag=True, help="Save the cleaned and consistent pre-processed JSON file")
@click.option("--use-json-input", "-j", is_flag=True, help="Accept JSON files as input")
@click.option("--debug", "-d", is_flag=True, help="Enable debug logging")
@click.option("--verbose", "-v", is_flag=True, help="Show all log messages in console")
@click.option("--start-hour", type=int, default=None, help="Hour offset for SRT timestamps")
@click.option("--setup", is_flag=True, help="Run setup wizard or configure defaults non-interactively")
@click.option("--api-key", help="Set API key for the specified API (requires --setup)")
@click.version_option()
@click.pass_context
def main(ctx, input_path, file, folder, api, language, output, chars_per_line, words_per_subtitle, 
         word_srt, davinci_srt, silent_portions, padding_start, padding_end, show_pauses, 
         filler_lines, filler_words, remove_fillers, speaker_labels, fps, fps_offset_start, 
         fps_offset_end, diarize, num_speakers, use_input, use_pcm, keep_flac, keep, model, 
         chunk_length, overlap, force, save_cleaned_json, use_json_input, debug, verbose, start_hour,
         setup, api_key):
    """Unified Audio Transcription Tool."""
    
    # If a subcommand is invoked, do nothing here (let the subcommand handle it)
    if ctx.invoked_subcommand is not None:
        return

    # Set up logging
    setup_logger(debug=debug, verbose=verbose)
    
    # Initialize ConfigManager
    config = ConfigManager()

    # Handle setup flag
    if setup:
        # Check if we have other arguments for non-interactive setup
        # We need to check the context or kwargs, but since we have them as args:
        non_interactive_args = False
        if api or language or output or model or api_key:
            non_interactive_args = True
        
        if non_interactive_args:
            logger.info("Running non-interactive setup...")
            
            # If API is specified, update default API
            if api:
                config.set("default_api", api)
                logger.success(f"Set default API to: {api}")
            
            # If API key is specified
            if api_key:
                # If API name is provided, use it, otherwise use default
                target_api = api or config.get("default_api")
                if target_api:
                    config.set_api_key(target_api, api_key)
                    logger.success(f"Set API key for {target_api}")
                else:
                    logger.error("Cannot set API key: No API specified and no default API set.")
            
            # Handle other defaults
            if language:
                config.set("default_language", language)
                logger.success(f"Set default language to: {language}")
                
            if model:
                # If API is specified, set model for that API
                target_api = api or config.get("default_api")
                if target_api:
                    config.set_model(target_api, model)
                    logger.success(f"Set default model for {target_api} to: {model}")
                else:
                    logger.warning("Model specified but no API context found. Skipping model default.")

            # Handle output formats
            if output:
                 # 'output' comes as a tuple from click
                 config.set("default_output_formats", list(output))
                 logger.success(f"Set default output formats to: {', '.join(output)}")

            logger.success("Configuration updated successfully.")
            sys.exit(0)
        else:
            # No other args, run interactive wizard
            run_setup_wizard()
            sys.exit(0)

    # Determine target path from positional arg or legacy flags
    target_path = input_path or file or folder

    # Determine if we should run in interactive mode
    should_run_interactive = False
    
    if not target_path:
        should_run_interactive = True
    
    if target_path and not api:
        should_run_interactive = True
        
    if should_run_interactive:
        logger.info("Entering interactive mode...")
        
        # Run interactive wizard
        options = run_interactive_mode(target_path)
        
        if not options:
            sys.exit(0)
            
        # Update local variables with options from wizard
        target_path = options.get("file") or options.get("folder")
        api = options.get("api")
        
        # Update other params if they exist in options
        if "model" in options: model = options["model"]
        if "language" in options: language = options["language"]
        if "output" in options: output = options["output"]
        if "remove_fillers" in options: remove_fillers = options["remove_fillers"]
        if "speaker_labels" in options: speaker_labels = options["speaker_labels"]
        if "diarize" in options: diarize = options["diarize"]
        if "num_speakers" in options: num_speakers = options["num_speakers"]
        if "davinci_srt" in options: davinci_srt = options["davinci_srt"]
        if "filler_lines" in options: filler_lines = options["filler_lines"]
        if "silent_portions" in options: silent_portions = options["silent_portions"]
        
    # If still no API set (and not interactive), load from config or default to groq
    if not api:
        api = config.get("default_api", "groq")
        logger.info(f"Using default API: {api}")

    # Validate input parameters for default action
    if not target_path:
        click.echo("Error: No file or folder specified.")
        sys.exit(1)
    
    # Handle "all" output format
    output_formats = list(output)
    if not output_formats: # If empty (e.g. from interactive mode default)
        output_formats = ["text", "srt"]
        
    if "all" in output_formats:
        output_formats = ["text", "srt", "word_srt", "davinci_srt", "json"]
    
    # Collect raw user params (only those that are not None)
    raw_user_params = {
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
        "keep": keep,
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
    
    # Determine preset
    preset = "davinci" if davinci_srt else None
    
    # Get effective parameters using DefaultsManager
    kwargs = DefaultsManager.get_effective_params(api, raw_user_params, preset=preset)
    
    # Ensure output is a list (DefaultsManager might return it as tuple if it came from defaults)
    if "output" in kwargs and not isinstance(kwargs["output"], list):
        kwargs["output"] = list(kwargs["output"])
    
    # Run processing
    # Check if target_path is a directory or file
    target_path_obj = Path(target_path)
    if target_path_obj.is_dir():
        process_audio_path(target_path, **kwargs)
    else:
        process_audio_path(target_path, **kwargs)

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