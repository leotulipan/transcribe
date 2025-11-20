"""
Unified output formatters for transcription results.

This module provides functions to convert transcription results into various output formats
like SRT, word-level SRT, Davinci SRT, and plain text.
"""
from typing import Union, List, Dict, Any, Optional, Tuple
from pathlib import Path
import os
from loguru import logger

from audio_transcribe.utils.parsers import TranscriptionResult


def create_text_file(result: TranscriptionResult, output_file: Union[str, Path]) -> None:
    """
    Create a plain text file from a TranscriptionResult.
    
    Args:
        result: TranscriptionResult object
        output_file: Path to the output text file
    """
    logger.info(f"Creating text file: {output_file}")
    
    # Extract words and build text
    text = result.text
    
    # If text is empty but we have words, build the text from the words array
    if not text.strip() and result.words:
        text = ""
        for word in result.words:
            if word.get('type') == 'spacing':
                # Add space or pause indicator
                if "(...)" in word.get('text', ''):
                    text += " (...) "
                else:
                    text += " "
            else:
                # Add word text
                word_text = word.get('word', word.get('text', ''))
                if word_text:
                    # Avoid double spaces
                    if text and not text.endswith(" ") and not text.endswith("(...) "):
                        text += " "
                    text += word_text
        # Trim any extra spaces
        text = text.strip()
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)
    
    logger.info(f"Text file created: {output_file}")


def create_srt_file(result: TranscriptionResult, output_file: Union[str, Path], 
                   format_type: str = "standard", **kwargs) -> None:
    """
    Create an SRT file from a TranscriptionResult.
    
    Args:
        result: TranscriptionResult object
        output_file: Path to the output SRT file
        format_type: Type of SRT format to create ("standard", "word", or "davinci")
        **kwargs: Additional format-specific parameters
    """
    logger.info(f"Creating SRT file ({format_type}): {output_file}")
    
    # Import from transcribe_helpers to reuse existing implementation
    from audio_transcribe.transcribe_helpers.output_formatters import create_srt
    
    # Apply any needed modifications to words
    words = result.words
    filler_lines = kwargs.get("filler_lines", False)
    filler_words = kwargs.get("filler_words", None)
    
    # Debug the silentportions/silent_portions parameter
    silent_portions = kwargs.get("silent_portions", kwargs.get("silentportions", 0))
    show_pauses = kwargs.get("show_pauses", False) or silent_portions > 0
    logger.debug(f"First 5 words: {words[:5] if words else 'None'}")
    logger.debug(f"Silent portions threshold: {silent_portions}ms")
    logger.debug(f"Show pauses: {show_pauses}")
    logger.debug(f"Converting timestamps for SRT formatting")
    
    # Look for pause markers in the data
    pause_count = 0
    for word in words[:50]:  # Check first 50 words
        if word.get('type') == 'spacing' and '(...)' in word.get('text', ''):
            pause_count += 1
            logger.debug(f"Found pause marker: {word}")
    logger.debug(f"Found {pause_count} pause markers in input words")
    
    # If filler_lines is requested, ensure we do not strip fillers earlier
    # and pass control flags down to the low-level create_srt
    start_hour = kwargs.get("start_hour") or 0
    if format_type == "word":
        create_srt(
            words, 
            output_file, 
            srt_mode="word",
            fps=kwargs.get("fps"),
            fps_offset_start=kwargs.get("fps_offset_start", -1),
            fps_offset_end=kwargs.get("fps_offset_end", 0),
            padding_start=kwargs.get("padding_start", 0),
            padding_end=kwargs.get("padding_end", 0),
            remove_fillers=kwargs.get("remove_fillers", False),
            filler_words=kwargs.get("filler_words"),
            show_pauses=show_pauses,
            silentportions=silent_portions,
            start_hour=start_hour
        )
    elif format_type == "davinci":
        create_srt(
            words, 
            output_file, 
            srt_mode="davinci",
            chars_per_line=kwargs.get("chars_per_line", 500),
            silentportions=silent_portions,
            fps=kwargs.get("fps"),
            fps_offset_start=kwargs.get("fps_offset_start", -1),
            fps_offset_end=kwargs.get("fps_offset_end", 0),
            padding_start=kwargs.get("padding_start", -125),
            padding_end=kwargs.get("padding_end", 0),
            remove_fillers=False if filler_lines else kwargs.get("remove_fillers", True),
            filler_words=filler_words,
            max_words_per_block=kwargs.get("max_words_per_block", 500),
            show_pauses=show_pauses,
            start_hour=start_hour,
            filler_lines=filler_lines
        )
    else:  # standard
        create_srt(
            words, 
            output_file, 
            srt_mode="standard",
            chars_per_line=kwargs.get("chars_per_line", 80),
            silentportions=silent_portions,
            fps=kwargs.get("fps"),
            fps_offset_start=kwargs.get("fps_offset_start", -1),
            fps_offset_end=kwargs.get("fps_offset_end", 0),
            padding_start=kwargs.get("padding_start", 0),
            padding_end=kwargs.get("padding_end", 0),
            remove_fillers=False if filler_lines else kwargs.get("remove_fillers", False),
            filler_words=filler_words,
            show_pauses=show_pauses,
            start_hour=start_hour,
            words_per_subtitle=kwargs.get("words_per_subtitle", 0),
            filler_lines=filler_lines
        )
    
    logger.info(f"SRT file created: {output_file}")


def create_output_files(result: TranscriptionResult, audio_path: Union[str, Path], 
                       format_types: List[str] = None, **kwargs) -> Dict[str, str]:
    """
    Create output files in various formats from a transcription result.
    
    Args:
        result: TranscriptionResult object
        audio_path: Path to the original audio file (used to determine output location)
        format_types: List of output formats to create (default: ["text", "srt"])
        **kwargs: Additional format-specific parameters
        
    Returns:
        Dictionary mapping format types to created file paths
    """
    if format_types is None:
        format_types = ["text", "srt"]
    
    file_path = Path(audio_path)
    file_dir = file_path.parent
    # Remove _apiname from stem if present
    file_name = file_path.stem
    for api in ["_assemblyai", "_elevenlabs", "_groq", "_openai"]:
        if file_name.endswith(api):
            file_name = file_name[: -len(api)]
    
    # Make sure silent portions are properly marked if requested
    # Use either silent_portions or silentportions parameter
    silent_portions = kwargs.get("silent_portions", kwargs.get("silentportions", 0))
    show_pauses = kwargs.get("show_pauses", False) or silent_portions > 0
    
    # Debug
    logger.debug(f"create_output_files - silent_portions={silent_portions}")
    logger.debug(f"create_output_files - show_pauses={show_pauses} (kwarg={kwargs.get('show_pauses', 'not set')})")
    
    # Process filler words first if requested (moved this section up)
    remove_fillers = kwargs.get("remove_fillers", False)
    if remove_fillers and result.words:
        from audio_transcribe.transcribe_helpers.text_processing import process_filler_words
        logger.debug("Processing filler words before pause detection")
        result.words = process_filler_words(
            result.words,
            silent_portions,
            kwargs.get("filler_words", None)
        )
    
    # Re-standardize word format to ensure appropriate spacing elements
    # for silent portions detection - MOVED OUTSIDE the format loop
    if silent_portions > 0 or show_pauses:
        from audio_transcribe.transcribe_helpers.text_processing import standardize_word_format
        logger.debug(f"Applying standardize_word_format with show_pauses={show_pauses}, silence_threshold={silent_portions}")
        result.words = standardize_word_format(
            result.words,
            show_pauses=show_pauses,
            silence_threshold=silent_portions
        )
        
        # Log some of the words to verify pauses were added
        pause_count = 0
        for word in result.words[:50]:  # Check first 50 words
            if word.get('type') == 'spacing' and '(...)' in word.get('text', ''):
                pause_count += 1
        logger.debug(f"Found {pause_count} pause markers in first 50 words after standardization")
    
    # Now just use result.words for all formats
    created_files = {}
    word_srt_flag = kwargs.get("word_srt", False)
    for format_type in format_types:
        if format_type == "srt":
            output_file = file_dir / f"{file_name}.srt"
            if word_srt_flag:
                create_srt_file(result, output_file, "word", **kwargs)
            else:
                create_srt_file(result, output_file, "standard", **kwargs)
            created_files["srt"] = str(output_file)
        elif format_type == "word_srt":
            output_file = file_dir / f"{file_name}.word.srt"
            create_srt_file(result, output_file, "word", **kwargs)
            created_files["word_srt"] = str(output_file)
        elif format_type == "davinci_srt":
            output_file = file_dir / f"{file_name}.davinci.srt"
            create_srt_file(result, output_file, "davinci", **kwargs)
            created_files["davinci_srt"] = str(output_file)
        elif format_type == "text":
            output_file = file_dir / f"{file_name}.txt"
            create_text_file(result, output_file)
            created_files["text"] = str(output_file)
        elif format_type == "json":
            output_file = file_dir / f"{file_name}.json"
            result.save(output_file)
            created_files["json"] = str(output_file)
    return created_files 