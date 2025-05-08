"""
Unified output formatters for transcription results.

This module provides functions to convert transcription results into various output formats
like SRT, word-level SRT, Davinci SRT, and plain text.
"""
from typing import Union, List, Dict, Any, Optional, Tuple
from pathlib import Path
import os
from loguru import logger

from utils.parsers import TranscriptionResult


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
    from transcribe_helpers.output_formatters import create_srt
    
    # Apply any needed modifications to words
    words = result.words
    
    # Debug the silentportions/silent_portions parameter
    silent_portions = kwargs.get("silent_portions", kwargs.get("silentportions", 0))
    print(f"DEBUG formatters.create_srt_file - silent_portions={silent_portions}")
    
    # Create SRT file using the appropriate format
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
            filler_words=kwargs.get("filler_words")
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
            remove_fillers=kwargs.get("remove_fillers", True),
            filler_words=kwargs.get("filler_words"),
            max_words_per_block=kwargs.get("max_words_per_block", 500)
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
            remove_fillers=kwargs.get("remove_fillers", False),
            filler_words=kwargs.get("filler_words")
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
    file_name = file_path.stem
    
    # Make sure silent portions are properly marked if requested
    # Use either silent_portions or silentportions parameter
    silent_portions = kwargs.get("silent_portions", kwargs.get("silentportions", 0))
    show_pauses = kwargs.get("show_pauses", False) or silent_portions > 0
    
    # Debug
    print(f"DEBUG formatters.create_output_files - silent_portions={silent_portions}")
    
    # Re-standardize word format to ensure appropriate spacing elements
    # for silent portions detection
    if (show_pauses or silent_portions > 0) and result.words:
        from transcribe_helpers.text_processing import standardize_word_format
        result.words = standardize_word_format(
            result.words, 
            show_pauses=show_pauses,
            silence_threshold=silent_portions
        )
    
    created_files = {}
    
    for format_type in format_types:
        if format_type == "text":
            output_file = file_dir / f"{file_name}.txt"
            create_text_file(result, output_file)
            created_files["text"] = str(output_file)
        
        elif format_type == "srt":
            output_file = file_dir / f"{file_name}.srt"
            # Pass both parameter names for backward compatibility
            create_srt_file(result, output_file, "standard", 
                           silentportions=silent_portions, 
                           **kwargs)
            created_files["srt"] = str(output_file)
        
        elif format_type == "word_srt":
            output_file = file_dir / f"{file_name}.word.srt"
            create_srt_file(result, output_file, "word", **kwargs)
            created_files["word_srt"] = str(output_file)
        
        elif format_type == "davinci_srt":
            output_file = file_dir / f"{file_name}.davinci.srt"
            # Pass both parameter names for backward compatibility
            create_srt_file(result, output_file, "davinci", 
                           silentportions=silent_portions,
                           **kwargs)
            created_files["davinci_srt"] = str(output_file)
        
        elif format_type == "json":
            output_file = file_dir / f"{file_name}.json"
            result.save(output_file)
            created_files["json"] = str(output_file)
    
    return created_files 