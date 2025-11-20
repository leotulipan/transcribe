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
        audio: AudioSegment to check
        file_extension: File extension for format-specific checks
        
    Returns:
        True if audio meets requirements, False otherwise
    
    From: elevenlabs - Check audio format requirements
    """
    # Check basic requirements
    if audio.channels != 1:
        return False
    if audio.frame_rate != 16000:
        return False
    if audio.sample_width != 2:  # 16-bit
        return False
    return True


def extract_audio_from_mp4(input_path: Union[str, Path]) -> Optional[str]:
    """
    Extract audio stream from MP4 file without re-encoding.
    
    Args:
        input_path: Path to MP4 file
        
    Returns:
        Path to extracted audio file (M4A) or None if extraction failed
    """
    if isinstance(input_path, str):
        input_path = Path(input_path)
        
    # Only process MP4 files
    if input_path.suffix.lower() != '.mp4':
        logger.debug(f"File is not MP4, skipping extraction: {input_path}")
        return None
        
    logger.info(f"Extracting audio from MP4: {input_path}")
    
    try:
        # Load the video file
        audio = AudioSegment.from_file(str(input_path))
        
        # Create output path with M4A extension
        output_path = input_path.with_suffix('.m4a')
        
        # Export audio without re-encoding (copy audio stream)
        audio.export(str(output_path), format="mp4", codec="aac")
        
        # Check if extraction was successful and resulted in smaller file
        original_size_mb = os.path.getsize(input_path) / (1024 * 1024)
        extracted_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        
        logger.info(f"Extracted audio: {extracted_size_mb:.1f}MB (original: {original_size_mb:.1f}MB)")
        
        if extracted_size_mb < original_size_mb:
            logger.info(f"Audio extraction successful, file size reduced by {original_size_mb - extracted_size_mb:.1f}MB")
            return str(output_path)
        else:
            logger.warning(f"Extracted audio is not smaller than original, removing extracted file")
            os.unlink(output_path)
            return None
            
    except Exception as e:
        logger.error(f"Audio extraction failed: {str(e)}")
        return None


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
        # https://elevenlabs.io/docs/api-reference/speech-to-text/convert 1GB
        return 1000
    return 25  # fallback


def convert_to_mp3(input_path: Union[str, Path], bitrate: str = "128k") -> Optional[str]:
    """
    Convert audio file to MP3 format with specified bitrate.
    
    Args:
        input_path: Path to input audio/video file
        bitrate: Target bitrate (e.g., "128k", "64k")
        
    Returns:
        Path to converted MP3 file or None if conversion failed
    """
    if isinstance(input_path, str):
        input_path = Path(input_path)
        
    # Create a temporary file with .mp3 extension
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
        output_path = temp_file.name
        
    logger.info(f"Converting {input_path} to MP3 ({bitrate})...")
    
    try:
        audio = AudioSegment.from_file(str(input_path))
        # Set channels to mono for speech optimization
        audio = audio.set_channels(1)
        # Export as MP3
        audio.export(output_path, format="mp3", bitrate=bitrate)
        
        logger.info(f"Converted to MP3: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Audio conversion to MP3 failed: {str(e)}")
        if os.path.exists(output_path):
            os.unlink(output_path)
        return None


def optimize_audio_for_api(input_path: Union[str, Path], api_name: str) -> Tuple[Path, bool]:
    """
    Optimize audio file to meet API file size limits using a cascade of strategies.
    
    Strategies:
    1. Check if original file fits.
    2. If video, extract audio (m4a/aac).
    3. Convert to FLAC (lossless compression).
    4. Convert to MP3 128kbps mono (lossy compression).
    
    Args:
        input_path: Path to the input file
        api_name: Name of the API to check limits for
        
    Returns:
        Tuple of (path_to_optimized_file, is_temporary_file)
    """
    input_path = Path(input_path)
    max_size_mb = get_api_file_size_limit(api_name)
    
    logger.info(f"Optimizing {input_path.name} for {api_name} (Limit: {max_size_mb}MB)...")
    
    # Track intermediate files to clean up
    intermediate_files = []
    
    current_path = input_path
    is_current_temp = False
    
    try:
        # 1. Check original file
        current_size_mb = os.path.getsize(current_path) / (1024 * 1024)
        if current_size_mb <= max_size_mb:
            logger.info(f"Original file fits ({current_size_mb:.2f}MB <= {max_size_mb}MB). No conversion needed.")
            return current_path, False
            
        logger.info(f"File too large ({current_size_mb:.2f}MB). Attempting optimization...")
        
        # 2. If video, extract audio
        video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.webm']
        if current_path.suffix.lower() in video_extensions:
            logger.info("Strategy 1: Extracting audio from video...")
            extracted_path = extract_audio_from_mp4(current_path)
            if extracted_path:
                extracted_path = Path(extracted_path)
                size_mb = os.path.getsize(extracted_path) / (1024 * 1024)
                
                # Update current path to the new file
                if is_current_temp:
                    intermediate_files.append(current_path)
                current_path = extracted_path
                is_current_temp = True
                
                if size_mb <= max_size_mb:
                    logger.success(f"Audio extraction successful. New size: {size_mb:.2f}MB")
                    _cleanup_intermediates(intermediate_files)
                    return current_path, True
                else:
                    logger.info(f"Extracted audio still too large ({size_mb:.2f}MB). Continuing to next strategy...")
        
        # 3. Convert to FLAC
        logger.info("Strategy 2: Converting to FLAC...")
        flac_path = convert_to_flac(current_path)
        if flac_path:
            flac_path = Path(flac_path)
            size_mb = os.path.getsize(flac_path) / (1024 * 1024)
            
            # Update current path
            if is_current_temp:
                intermediate_files.append(current_path)
            current_path = flac_path
            is_current_temp = True
            
            if size_mb <= max_size_mb:
                logger.success(f"FLAC conversion successful. New size: {size_mb:.2f}MB")
                _cleanup_intermediates(intermediate_files)
                return current_path, True
            else:
                logger.info(f"FLAC file still too large ({size_mb:.2f}MB). Continuing to next strategy...")
                
        # 4. Convert to MP3 128kbps mono
        logger.info("Strategy 3: Converting to MP3 (128k mono)...")
        mp3_path = convert_to_mp3(current_path, bitrate="128k")
        if mp3_path:
            mp3_path = Path(mp3_path)
            size_mb = os.path.getsize(mp3_path) / (1024 * 1024)
            
            # Update current path
            if is_current_temp:
                intermediate_files.append(current_path)
            current_path = mp3_path
            is_current_temp = True
            
            if size_mb <= max_size_mb:
                logger.success(f"MP3 conversion successful. New size: {size_mb:.2f}MB")
                _cleanup_intermediates(intermediate_files)
                return current_path, True
            else:
                logger.error(f"MP3 file still too large ({size_mb:.2f}MB). Optimization failed.")
                # If this failed, we might as well return this one or fail completely?
                # But the function contract implies we return something usable or raise.
                # Let's clean up this last attempt since it failed.
                if mp3_path.exists():
                    os.unlink(mp3_path)
                # Revert current_path to previous valid one (though previous ones were also too big)
                # We will fall through to raise RuntimeError
                
        # Clean up all intermediates if we failed entirely
        _cleanup_intermediates(intermediate_files)
        if is_current_temp and current_path.exists():
             # If we ended up with a temp file that is still too big, and we are giving up
             try:
                 os.unlink(current_path)
             except:
                 pass

        raise RuntimeError(f"Could not optimize file to under {max_size_mb}MB. Please split the file manually.")
        
    except Exception as e:
        # Clean up on error
        _cleanup_intermediates(intermediate_files)
        if is_current_temp and current_path.exists():
            try:
                os.unlink(current_path)
            except:
                pass
        raise e

def _cleanup_intermediates(files: List[Path]):
    """Helper to clean up intermediate files."""
    for f in files:
        if f.exists():
            try:
                os.unlink(f)
            except Exception as e:
                logger.warning(f"Failed to delete intermediate file {f}: {e}")
