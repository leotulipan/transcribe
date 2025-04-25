"""
Utility functions for audio transcription
"""
import os
from pathlib import Path
from typing import Optional, Union
from loguru import logger


def setup_logger(debug: bool = False, verbose: bool = False) -> None:
    """
    Configure loguru logger with appropriate log levels.
    
    Args:
        debug: Enable debug mode for maximum verbosity
        verbose: Enable info-level messages
    
    From: elevenlabs - Configure loguru logger with levels
    """
    logger.remove()  # Remove default handler
    
    # Add file logging
    logger.add(
        "transcribe_{time}.log",
        rotation="1 day",
        retention="7 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    
    # Configure console logging based on verbosity
    if debug:
        logger.add(lambda msg: print(msg), level="DEBUG")
    elif verbose:
        logger.add(lambda msg: print(msg), level="INFO")
    else:
        logger.add(lambda msg: print(msg), level="ERROR")


def in_debug_mode(debug_flag: bool = False) -> bool:
    """
    Check if debug mode is enabled.
    
    Args:
        debug_flag: Debug flag from command line arguments
        
    Returns:
        True if debug mode is enabled
    
    From: multiple - Check if debug mode enabled
    """
    return debug_flag


def check_transcript_exists(file_path: Union[str, Path], file_name: str) -> bool:
    """
    Check if a transcript already exists for the given file.
    
    Args:
        file_path: Directory containing the audio file
        file_name: Name of the audio file without extension
        
    Returns:
        True if transcript exists, False otherwise
    
    From: multiple - Check for existing transcript files
    """
    json_path = os.path.join(file_path, f"{file_name}.json")
    txt_path = os.path.join(file_path, f"{file_name}.txt")
    srt_path = os.path.join(file_path, f"{file_name}.srt")
    
    return os.path.exists(json_path) or os.path.exists(txt_path) or os.path.exists(srt_path)


def min_timestamp(ts1: str, ts2: str) -> str:
    """
    Return the earlier of two timestamps in SRT format (HH:MM:SS,mmm).
    
    Args:
        ts1: First timestamp in SRT format
        ts2: Second timestamp in SRT format
        
    Returns:
        Earlier timestamp
    
    From: elevenlabs - Return earlier of two timestamps
    """
    h1, m1, rest1 = ts1.split(':')
    s1, ms1 = rest1.split(',')
    
    h2, m2, rest2 = ts2.split(':')
    s2, ms2 = rest2.split(',')
    
    time1 = int(h1) * 3600 + int(m1) * 60 + int(s1) + int(ms1) / 1000.0
    time2 = int(h2) * 3600 + int(m2) * 60 + int(s2) + int(ms2) / 1000.0
    
    return ts1 if time1 <= time2 else ts2


def max_timestamp(ts1: str, ts2: str) -> str:
    """
    Return the later of two timestamps in SRT format (HH:MM:SS,mmm).
    
    Args:
        ts1: First timestamp in SRT format
        ts2: Second timestamp in SRT format
        
    Returns:
        Later timestamp
    
    From: elevenlabs - Return later of two timestamps
    """
    h1, m1, rest1 = ts1.split(':')
    s1, ms1 = rest1.split(',')
    
    h2, m2, rest2 = ts2.split(':')
    s2, ms2 = rest2.split(',')
    
    time1 = int(h1) * 3600 + int(m1) * 60 + int(s1) + int(ms1) / 1000.0
    time2 = int(h2) * 3600 + int(m2) * 60 + int(s2) + int(ms2) / 1000.0
    
    return ts1 if time1 >= time2 else ts2


def save_results(result: dict, audio_path: Union[str, Path], language: str = "en") -> Path:
    """
    Save transcription results to multiple files (txt, json, srt).
    
    Args:
        result: Transcription result dictionary
        audio_path: Path to the original audio file
        language: Language code for the transcript
        
    Returns:
        Base path where files were saved
    
    From: groq - Save transcript in multiple formats
    """
    from datetime import datetime
    import json
    from pathlib import Path
    from .output_formatters import convert_to_srt
    
    try:
        output_dir = Path("transcriptions")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_path = Path(audio_path)
        base_path = output_dir / f"{audio_path.stem}_{timestamp}"
        
        # Save results in different formats
        with open(f"{base_path}.{language}.txt", 'w', encoding='utf-8') as f:
            f.write(result["text"])
        
        with open(f"{base_path}_full.{language}.json", 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        if "segments" in result:
            with open(f"{base_path}_segments.{language}.json", 'w', encoding='utf-8') as f:
                json.dump(result["segments"], f, indent=2, ensure_ascii=False)
        
        # Convert to SRT format
        convert_to_srt(result, base_path)
        
        logger.info(f"Results saved to transcriptions folder:")
        logger.info(f"- {base_path}.{language}.txt")
        logger.info(f"- {base_path}_full.{language}.json")
        if "segments" in result:
            logger.info(f"- {base_path}_segments.{language}.json")
        logger.info(f"- {base_path}.{language}.srt")
        
        return base_path
    
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        raise
