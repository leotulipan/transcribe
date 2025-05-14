"""
Audio processing functions for transcription
"""
import os
import tempfile
import subprocess
import base64
from pathlib import Path
from typing import Union, Optional, List, Dict, Any, Tuple
from pydub import AudioSegment
from loguru import logger

# Try to import loguru, fallback to our mock implementation
try:
    from loguru import logger
except ImportError:
    import sys
    import os
    
    # Add the parent directory to sys.path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Import our mock logger
    from loguru_patch import logger


def check_audio_length(file_path: Union[str, Path], max_length: int = 7200) -> bool:
    """
    Check if audio file is shorter than maximum allowed length.
    
    Args:
        file_path: Path to the audio file
        max_length: Maximum allowed length in seconds
        
    Returns:
        True if audio is within limit, raises RuntimeError otherwise
    
    From: elevenlabs - Check if audio duration exceeds limit
    """
    audio = AudioSegment.from_file(file_path)
    duration_seconds = len(audio) / 1000.0  # Convert milliseconds to seconds
    if duration_seconds > max_length:
        raise RuntimeError(f"Audio duration ({duration_seconds:.1f}s) exceeds maximum allowed length ({max_length}s)")
    return True


def check_audio_format(audio: AudioSegment, file_extension: str = None) -> bool:
    """
    Check if audio meets requirements (mono, 16kHz, 16-bit).
    
    Args:
        audio: AudioSegment object to check
        file_extension: Optional file extension to determine if format checks should be bypassed
        
    Returns:
        True if audio meets requirements or is FLAC (bypass), False otherwise
    
    From: elevenlabs - Verify audio has correct specifications
    """
    # Always return True for FLAC files to bypass format checks
    if file_extension is not None and file_extension.lower() == '.flac':
        return True
        
    return (audio.channels == 1 and 
            audio.frame_rate == 16000 and 
            audio.sample_width == 2)


def convert_to_flac(input_path: Union[str, Path], sample_rate: int = 16000) -> Optional[str]:
    """
    Convert audio file to 16kHz mono FLAC format for API processing.
    
    Args:
        input_path: Path to input audio/video file
        sample_rate: Target sample rate (16kHz for most speech APIs)
        
    Returns:
        Path to converted FLAC file or None if conversion failed
    """
    if isinstance(input_path, str):
        input_path = Path(input_path)
        
    # If already a FLAC file, just return the path
    if input_path.suffix.lower() == '.flac':
        logger.info(f"File is already in FLAC format: {input_path}")
        return str(input_path)
        
    # Create a temporary file with .flac extension
    with tempfile.NamedTemporaryFile(suffix='.flac', delete=False) as temp_file:
        output_path = temp_file.name
        
    logger.info(f"Converting {input_path} to 16kHz mono FLAC...")
    
    try:
        # Use pydub for conversion instead of FFmpeg
        audio = AudioSegment.from_file(str(input_path))
        # Set frame rate to specified sample rate and channels to mono
        audio = audio.set_frame_rate(sample_rate).set_channels(1)
        # Export as FLAC
        audio.export(output_path, format="flac")
        
        logger.info(f"Converted to FLAC: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Audio conversion failed: {str(e)}")
        if os.path.exists(output_path):
            os.unlink(output_path)
        return None


def convert_to_pcm(input_file: Union[str, Path]) -> Path:
    """
    Convert audio/video file to PCM format (mono, 16-bit, 16kHz).
    
    Args:
        input_file: Path to input audio/video file
        
    Returns:
        Path to converted PCM WAV file
    
    From: elevenlabs - Convert audio to PCM format
    """
    logger.info(f"Converting {input_file} to PCM format...")
    audio = AudioSegment.from_file(input_file)
    # Convert to mono
    audio = audio.set_channels(1)
    # Set sample rate to 16kHz
    audio = audio.set_frame_rate(16000)
    # Set sample width to 2 bytes (16-bit)
    audio = audio.set_sample_width(2)
    
    # Create output filename
    output_file = os.path.splitext(input_file)[0] + "_converted.wav"
    # Export as PCM WAV
    audio.export(output_file, format="wav", parameters=["-f", "s16le"])
    logger.info(f"PCM conversion completed: {output_file}")
    return Path(output_file)


def check_file_size(file_path: Union[str, Path], max_size_mb: int = 1000) -> bool:
    """
    Check if file size is under the specified limit.
    
    Args:
        file_path: Path to the file to check
        max_size_mb: Maximum allowed size in MB
        
    Returns:
        True if file is within limit, raises RuntimeError otherwise
    
    From: elevenlabs - Check if file exceeds size limit
    """
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise RuntimeError(f"File size ({size_mb:.2f}MB) exceeds {max_size_mb}MB limit")
    return True


def preprocess_audio(input_path: Union[str, Path]) -> Path:
    """
    Preprocess audio file to 16kHz mono FLAC using pydub.
    
    Args:
        input_path: Path to input audio file
        
    Returns:
        Path to preprocessed audio file
    
    From: groq - Convert audio to optimal format
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    with tempfile.NamedTemporaryFile(suffix='.flac', delete=False) as temp_file:
        output_path = Path(temp_file.name)
        
    logger.info(f"Converting {input_path} to 16kHz mono FLAC using pydub...")
    try:
        # Load audio file
        audio = AudioSegment.from_file(str(input_path))
        # Set frame rate to 16kHz and channels to mono
        audio = audio.set_frame_rate(16000).set_channels(1)
        # Export as FLAC
        audio.export(str(output_path), format="flac")
        logger.info(f"Converted successfully to: {output_path}")
        return output_path
    except Exception as e:
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        logger.error(f"Audio conversion failed: {e}")
        raise RuntimeError(f"Audio conversion failed: {e}")


def preprocess_audio_with_ffmpeg(input_path: Union[str, Path]) -> Path:
    """
    Preprocess audio file to 16kHz mono FLAC using pydub.
    This function is kept for backward compatibility but now uses the pydub implementation.
    
    Args:
        input_path: Path to input audio file
        
    Returns:
        Path to preprocessed audio file
    
    From: groq - Preprocess audio with ffmpeg (now using pydub)
    """
    logger.info("Using pydub-based audio preprocessing (FFmpeg no longer used directly)...")
    return preprocess_audio(input_path)


def audio_to_base64(file_path: Union[str, Path]) -> str:
    """
    Convert audio file to base64 string.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Base64 encoded string of audio file
    
    From: diarization - Convert audio file to base64
    """
    try:
        with open(file_path, "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode('utf-8')
    except Exception as e:
        raise RuntimeError(f"Failed to convert audio to base64: {e}")


def get_api_file_size_limit(api_name: str) -> int:
    """
    Return the max file size in MB for the given API.
    """
    api_name = api_name.lower()
    if api_name == "assemblyai":
        return 200
    if api_name == "groq":
        return 25
    if api_name == "openai":
        return 25
    if api_name == "elevenlabs":
        return 100
    return 25  # fallback
