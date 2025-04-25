"""
Audio processing functions for transcription
"""
import os
import tempfile
import subprocess
import base64
from pathlib import Path
from typing import Union, Optional
from pydub import AudioSegment
from loguru import logger


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


def check_audio_format(audio: AudioSegment) -> bool:
    """
    Check if audio meets requirements (mono, 16kHz, 16-bit).
    
    Args:
        audio: AudioSegment object to check
        
    Returns:
        True if audio meets requirements, False otherwise
    
    From: elevenlabs - Verify audio has correct specifications
    """
    return (audio.channels == 1 and 
            audio.frame_rate == 16000 and 
            audio.sample_width == 2)


def convert_to_flac(input_file: Union[str, Path]) -> Path:
    """
    Convert audio/video file to FLAC format (mono, 16-bit, 16kHz).
    
    Args:
        input_file: Path to input audio/video file
        
    Returns:
        Path to converted FLAC file
    
    From: elevenlabs - Convert audio to mono FLAC
    """
    logger.info(f"Converting {input_file} to FLAC format...")
    audio = AudioSegment.from_file(input_file)
    # Convert to mono
    audio = audio.set_channels(1)
    # Set sample rate to 16kHz
    audio = audio.set_frame_rate(16000)
    # Set sample width to 2 bytes (16-bit)
    audio = audio.set_sample_width(2)
    
    # Create output filename
    output_file = os.path.splitext(input_file)[0] + "_converted.flac"
    # Export as FLAC
    audio.export(output_file, format="flac")
    logger.info(f"FLAC conversion completed: {output_file}")
    return Path(output_file)


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
        
    logger.info("Converting audio to 16kHz mono FLAC...")
    try:
        # Load audio file
        audio = AudioSegment.from_file(input_path)
        # Set frame rate to 16kHz and channels to mono
        audio = audio.set_frame_rate(16000).set_channels(1)
        # Export as FLAC
        audio.export(output_path, format="flac")
        return output_path
    except Exception as e:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"Audio conversion failed: {e}")


def preprocess_audio_with_ffmpeg(input_path: Union[str, Path]) -> Path:
    """
    Preprocess audio file to 16kHz mono FLAC using ffmpeg.
    
    Args:
        input_path: Path to input audio file
        
    Returns:
        Path to preprocessed audio file
    
    From: groq - Preprocess audio with ffmpeg
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    with tempfile.NamedTemporaryFile(suffix='.flac', delete=False) as temp_file:
        output_path = Path(temp_file.name)
        
    logger.info("Converting audio to 16kHz mono FLAC using ffmpeg...")
    try:
        subprocess.run([
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'error',
            '-i', str(input_path),
            '-ar', '16000',
            '-ac', '1',
            '-c:a', 'flac',
            '-y',
            str(output_path)
        ], check=True) 
        return output_path
    except subprocess.CalledProcessError as e:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"FFmpeg conversion failed: {e.stderr}")


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
