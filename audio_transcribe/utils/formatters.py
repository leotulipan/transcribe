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
            silentportions=kwargs.get("silentportions", 250),
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
            silentportions=kwargs.get("silentportions", 0),
            fps=kwargs.get("fps"),
            fps_offset_start=kwargs.get("fps_offset_start", -1),
            fps_offset_end=kwargs.get("fps_offset_end", 0),
            padding_start=kwargs.get("padding_start", 0),
            padding_end=kwargs.get("padding_end", 0),
            remove_fillers=kwargs.get("remove_fillers", False),
            filler_words=kwargs.get("filler_words"),
            words_per_subtitle=kwargs.get("words_per_subtitle", 0)
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
    
    created_files = {}
    
    for format_type in format_types:
        if format_type == "text":
            output_file = file_dir / f"{file_name}.txt"
            create_text_file(result, output_file)
            created_files["text"] = str(output_file)
        
        elif format_type == "srt":
            output_file = file_dir / f"{file_name}.srt"
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
        
        elif format_type == "json":
            output_file = file_dir / f"{file_name}.json"
            result.save(output_file)
            created_files["json"] = str(output_file)
    
    return created_files 