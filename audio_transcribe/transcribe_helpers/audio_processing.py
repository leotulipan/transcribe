"""
Audio processing functions for transcription
"""
import os
import tempfile
import subprocess
import base64
import warnings
from pathlib import Path
from typing import Union, Optional, List, Dict, Any, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field

# Suppress pydub warnings (SyntaxWarning from invalid escapes, RuntimeWarning from ffmpeg check)
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pydub")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pydub")

from pydub import AudioSegment
from loguru import logger

# Import centralized ffmpeg utilities
from audio_transcribe.utils.ffmpeg import require_ffmpeg

# Import new modules
from .pyav_backend import (
    is_pyav_available,
    get_duration_seconds,
    extract_audio_pyav,
    convert_to_flac_pyav,
    convert_to_mp3_pyav,
    _get_duration_ffprobe
)

if TYPE_CHECKING:
    from .intermediate_files import IntermediateFileManager


# API format requirements for passthrough logic
API_FORMAT_REQUIREMENTS = {
    "groq": {"requires_flac": True, "accepts_video": False},
    "openai": {"requires_flac": True, "accepts_video": False},
    "assemblyai": {"requires_flac": False, "accepts_video": True},
    "elevenlabs": {"requires_flac": False, "accepts_video": True},
}


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


def _get_audio_duration_seconds(audio_path: Path) -> float:
    """
    Get audio/video duration in seconds using PyAV (10x faster) or ffprobe/pydub fallback.

    Args:
        audio_path: Path to audio/video file

    Returns:
        Duration in seconds, or 0.0 if unable to determine
    """
    # Try PyAV first (fastest)
    duration = get_duration_seconds(audio_path)
    if duration > 0:
        return duration

    # Fallback to pydub
    try:
        audio = AudioSegment.from_file(str(audio_path))
        return len(audio) / 1000.0
    except Exception:
        return 0.0

def _run_ffmpeg_with_progress(cmd: List[str], duration_seconds: float, operation_name: str) -> bool:
    """
    Run ffmpeg command with progress reporting.
    
    Args:
        cmd: ffmpeg command as list
        duration_seconds: Expected duration of media (for progress calculation)
        operation_name: Name of operation for logging
        
    Returns:
        True if successful, False otherwise
    """
    import time
    import re
    
    start_time = time.time()
    last_log_time = start_time
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Parse stderr for progress (ffmpeg outputs progress to stderr)
        while True:
            line = process.stderr.readline()
            if not line:
                break
                
            # Try to extract time from ffmpeg output
            # Format: "time=00:01:23.45" or "out_time_ms=83500"
            time_match = re.search(r'time=(\d+):(\d+):(\d+\.\d+)', line)
            if time_match:
                hours, minutes, seconds = time_match.groups()
                elapsed_seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                
                # Log progress every 2 seconds
                current_time = time.time()
                if current_time - last_log_time >= 2.0:
                    if duration_seconds > 0:
                        progress_pct = min(100, (elapsed_seconds / duration_seconds) * 100)
                        elapsed_str = f"{int(elapsed_seconds // 60):02d}:{int(elapsed_seconds % 60):02d}"
                        total_str = f"{int(duration_seconds // 60):02d}:{int(duration_seconds % 60):02d}"
                        logger.info(f"{operation_name}... {progress_pct:.0f}% ({elapsed_str}/{total_str})")
                    else:
                        elapsed_str = f"{int(elapsed_seconds // 60):02d}:{int(elapsed_seconds % 60):02d}"
                        logger.info(f"{operation_name}... {elapsed_str}")
                    last_log_time = current_time
        
        process.wait()
        return process.returncode == 0
        
    except Exception as e:
        logger.error(f"Error running ffmpeg: {e}")
        return False

def extract_audio_from_mp4(input_path: Union[str, Path]) -> Optional[str]:
    """
    Extract audio stream from MP4 file, preferring ffmpeg for speed.
    
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
        
    logger.info(f"Extracting audio from MP4: {input_path.name}")
    
    # Get duration for progress tracking
    duration_seconds = _get_audio_duration_seconds(input_path)
    
    # Create output path with M4A extension
    output_path = input_path.with_suffix('.m4a')
    
    # Try ffmpeg first (faster, can copy stream without re-encoding)
    try:
        # Ensure ffmpeg is available
        require_ffmpeg()

        # First try to copy audio stream directly (fastest)
        cmd = [
            'ffmpeg',
            '-i', str(input_path),
            '-vn',  # No video
            '-acodec', 'copy',  # Copy audio codec (no re-encoding)
            '-y',  # Overwrite output
            str(output_path)
        ]
        
        if _run_ffmpeg_with_progress(cmd, duration_seconds, "Extracting audio"):
            # Check if extraction was successful and resulted in smaller file
            if output_path.exists():
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
    except FileNotFoundError:
        # ffmpeg not available, fall back to pydub
        logger.debug("ffmpeg not found, using pydub for extraction")
    except Exception as e:
        logger.warning(f"ffmpeg extraction failed: {e}, falling back to pydub")
    
    # Fallback to pydub
    try:
        logger.info("Extracting audio using pydub (this may take longer)...")
        # Load the video file
        audio = AudioSegment.from_file(str(input_path))
        
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
        if output_path.exists():
            try:
                os.unlink(output_path)
            except:
                pass
        return None


def convert_to_flac(input_path: Union[str, Path], sample_rate: int = 16000) -> Optional[str]:
    """
    Convert audio file to 16kHz mono FLAC format for API processing.
    Prefers ffmpeg for speed, falls back to pydub.
    
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
        logger.info(f"File is already in FLAC format: {input_path.name}")
        return str(input_path)
        
    # Create a temporary file with .flac extension
    with tempfile.NamedTemporaryFile(suffix='.flac', delete=False) as temp_file:
        output_path = temp_file.name
        
    logger.info(f"Converting {input_path.name} to 16kHz mono FLAC...")

    # Get duration for progress tracking
    duration_seconds = _get_audio_duration_seconds(input_path)

    # Try ffmpeg first (faster)
    try:
        require_ffmpeg()

        cmd = [
            'ffmpeg',
            '-i', str(input_path),
            '-ar', str(sample_rate),
            '-ac', '1',  # Mono
            '-c:a', 'flac',
            '-y',
            output_path
        ]
        
        if _run_ffmpeg_with_progress(cmd, duration_seconds, "Converting to FLAC"):
            if os.path.exists(output_path):
                logger.info(f"Converted to FLAC: {output_path}")
                return output_path
    except FileNotFoundError:
        logger.debug("ffmpeg not found, using pydub for conversion")
    except Exception as e:
        logger.warning(f"ffmpeg conversion failed: {e}, falling back to pydub")
    
    # Fallback to pydub
    try:
        # Use pydub for conversion
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


@dataclass
class OptimizationResult:
    """Result of audio optimization with metadata for chunking decisions."""
    path: Path
    is_temporary: bool
    size_mb: float
    bytes_per_second: float  # Calculated from file size and audio duration
    intermediate_manager: Optional['IntermediateFileManager'] = field(default=None, compare=False, repr=False)

    def fits_limit(self, max_size_mb: float) -> bool:
        """Check if the optimized file fits within the size limit."""
        return self.size_mb <= max_size_mb

    def cleanup(self) -> None:
        """Clean up intermediate files via the intermediate manager."""
        if self.intermediate_manager:
            self.intermediate_manager.cleanup()


def can_passthrough(file_path: Path, api_name: str, threshold_mb: float) -> bool:
    """
    Check if file can skip processing entirely (direct upload).

    Args:
        file_path: Path to the file to check
        api_name: Name of the API to check against
        threshold_mb: Size threshold for passthrough

    Returns:
        True if file can be uploaded directly, False otherwise
    """
    size_mb = file_path.stat().st_size / (1024 * 1024)
    api_limit = get_api_file_size_limit(api_name)

    # Check file size against both threshold and API limit
    if size_mb > min(threshold_mb, api_limit):
        return False

    reqs = API_FORMAT_REQUIREMENTS.get(api_name.lower(), {"accepts_video": True})
    ext = file_path.suffix.lower()

    # Video files can only passthrough to APIs that accept them
    if ext in ['.mp4', '.mkv', '.avi', '.mov', '.webm']:
        return reqs.get("accepts_video", False)

    # FLAC-required APIs need FLAC format
    if reqs.get("requires_flac") and ext != '.flac':
        return False

    return True

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
    Prefers ffmpeg for speed, falls back to pydub.
    
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
        
    logger.info(f"Converting {input_path.name} to MP3 ({bitrate})...")

    # Get duration for progress tracking
    duration_seconds = _get_audio_duration_seconds(input_path)

    # Try ffmpeg first (faster)
    try:
        require_ffmpeg()

        cmd = [
            'ffmpeg',
            '-i', str(input_path),
            '-ac', '1',  # Mono
            '-b:a', bitrate,
            '-y',
            output_path
        ]
        
        if _run_ffmpeg_with_progress(cmd, duration_seconds, "Converting to MP3"):
            if os.path.exists(output_path):
                logger.info(f"Converted to MP3: {output_path}")
                return output_path
    except FileNotFoundError:
        logger.debug("ffmpeg not found, using pydub for conversion")
    except Exception as e:
        logger.warning(f"ffmpeg conversion failed: {e}, falling back to pydub")
    
    # Fallback to pydub
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


def _calculate_bytes_per_second(audio_path: Path) -> float:
    """
    Calculate bytes per second for an audio file.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Bytes per second (file_size_bytes / duration_seconds)
    """
    try:
        audio = AudioSegment.from_file(str(audio_path))
        duration_seconds = len(audio) / 1000.0  # Convert milliseconds to seconds
        file_size_bytes = os.path.getsize(audio_path)
        if duration_seconds > 0:
            return file_size_bytes / duration_seconds
        return 0.0
    except Exception as e:
        logger.warning(f"Failed to calculate bytes per second for {audio_path}: {e}")
        return 0.0

def optimize_audio_for_api(
    input_path: Union[str, Path],
    api_name: str,
    size_threshold_mb: float = 100.0,
    preserve_on_error: bool = True
) -> OptimizationResult:
    """
    Optimize audio file to meet API file size limits using a cascade of strategies.

    New algorithm:
    1. Check passthrough - if file is small and format compatible, return original
    2. Extract audio (if video) - use PyAV with ffmpeg fallback, check extracted size
    3. Convert to FLAC (if needed) - only if API requires FLAC or file still too large
    4. Convert to MP3 (last resort) - only if FLAC still exceeds limit
    5. Return with IntermediateFileManager attached for deferred cleanup

    Args:
        input_path: Path to the input file
        api_name: Name of the API to check limits for
        size_threshold_mb: Size threshold for passthrough (default 100MB)
        preserve_on_error: Whether to preserve intermediate files on error (default True)

    Returns:
        OptimizationResult with path, is_temporary, size_mb, bytes_per_second, and intermediate_manager
    """
    from .intermediate_files import IntermediateFileManager, FileOperation

    input_path = Path(input_path)
    max_size_mb = get_api_file_size_limit(api_name)

    logger.info(f"Optimizing {input_path.name} for {api_name} (Limit: {max_size_mb}MB)...")

    # Setup intermediate file manager
    manager = IntermediateFileManager(input_path)
    manager.setup()

    current_path = input_path
    is_current_temp = False
    original_size_mb = os.path.getsize(input_path) / (1024 * 1024)

    try:
        # 1. Check passthrough (new)
        if can_passthrough(current_path, api_name, size_threshold_mb):
            logger.info(f"Passthrough: File can be uploaded directly "
                       f"({original_size_mb:.2f}MB <= threshold {size_threshold_mb}MB)")
            bytes_per_sec = _calculate_bytes_per_second(current_path)
            return OptimizationResult(
                path=current_path,
                is_temporary=False,
                size_mb=original_size_mb,
                bytes_per_second=bytes_per_sec,
                intermediate_manager=None
            )

        # File needs processing
        logger.info(f"File requires processing ({original_size_mb:.2f}MB)")

        # 2. Extract audio (if video)
        video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.webm']
        if current_path.suffix.lower() in video_extensions:
            logger.info("Strategy 1: Extracting audio from video...")

            # Use intermediate manager for output path
            extracted_path = manager.get_path_for(FileOperation.EXTRACTED, current_path.suffix[1:])

            # Try PyAV first (faster)
            extracted_success = False
            if is_pyav_available():
                logger.debug("Using PyAV for audio extraction")
                extracted_success = extract_audio_pyav(current_path, extracted_path)

            # Fallback to ffmpeg subprocess
            if not extracted_success:
                logger.debug("PyAV extraction failed or unavailable, using ffmpeg subprocess")
                require_ffmpeg()  # Ensure ffmpeg is available before use

                # Use the existing extract_audio_from_mp4 but with our output path
                # We need to temporarily override the output path logic
                # For now, use a simpler approach with ffmpeg
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.m4a', delete=False) as temp_file:
                    temp_extracted = Path(temp_file.name)

                try:
                    # Use ffmpeg subprocess
                    cmd = ['ffmpeg', '-i', str(current_path), '-vn', '-acodec', 'copy', '-y', str(temp_extracted)]
                    result = subprocess.run(cmd, capture_output=True, timeout=120)
                    if result.returncode == 0 and temp_extracted.exists():
                        # Move to our managed path
                        import shutil
                        shutil.move(str(temp_extracted), str(extracted_path))
                        extracted_success = True
                except Exception as e:
                    logger.warning(f"ffmpeg extraction failed: {e}")
                finally:
                    if temp_extracted.exists():
                        try:
                            temp_extracted.unlink()
                        except:
                            pass

            if extracted_success and extracted_path.exists():
                size_mb = os.path.getsize(extracted_path) / (1024 * 1024)
                manager.register(extracted_path, FileOperation.EXTRACTED, current_path)

                # Check if extracted fits API limit
                reqs = API_FORMAT_REQUIREMENTS.get(api_name.lower(), {"requires_flac": True})
                if size_mb <= max_size_mb and not reqs.get("requires_flac"):
                    logger.success(f"Audio extraction successful. New size: {size_mb:.2f}MB")
                    bytes_per_sec = _calculate_bytes_per_second(extracted_path)
                    return OptimizationResult(
                        path=extracted_path,
                        is_temporary=True,
                        size_mb=size_mb,
                        bytes_per_second=bytes_per_sec,
                        intermediate_manager=manager
                    )
                else:
                    logger.info(f"Extracted audio: {size_mb:.2f}MB. Continuing to FLAC conversion...")
                    current_path = extracted_path
                    is_current_temp = True
            else:
                logger.warning("Audio extraction failed, continuing with original file")

        # 3. Convert to FLAC (if needed)
        reqs = API_FORMAT_REQUIREMENTS.get(api_name.lower(), {"requires_flac": True})
        current_size_mb = os.path.getsize(current_path) / (1024 * 1024)

        if reqs.get("requires_flac") or current_path.suffix.lower() != '.flac':
            if current_size_mb > max_size_mb or reqs.get("requires_flac"):
                logger.info("Strategy 2: Converting to FLAC...")

                flac_path = manager.get_path_for(FileOperation.CONVERTED_FLAC, "flac")

                # Try PyAV first
                flac_success = False
                if is_pyav_available():
                    logger.debug("Using PyAV for FLAC conversion")
                    flac_success = convert_to_flac_pyav(current_path, flac_path)

                # Fallback to ffmpeg subprocess via existing function
                if not flac_success:
                    logger.debug("PyAV FLAC conversion failed, using ffmpeg fallback")
                    flac_temp = convert_to_flac(current_path)
                    if flac_temp:
                        import shutil
                        shutil.move(str(flac_temp), str(flac_path))
                        flac_success = True

                if flac_success and flac_path.exists():
                    size_mb = os.path.getsize(flac_path) / (1024 * 1024)
                    manager.register(flac_path, FileOperation.CONVERTED_FLAC, current_path)

                    if size_mb <= max_size_mb:
                        logger.success(f"FLAC conversion successful. New size: {size_mb:.2f}MB")
                        bytes_per_sec = _calculate_bytes_per_second(flac_path)
                        return OptimizationResult(
                            path=flac_path,
                            is_temporary=True,
                            size_mb=size_mb,
                            bytes_per_second=bytes_per_sec,
                            intermediate_manager=manager
                        )
                    else:
                        logger.info(f"FLAC file: {size_mb:.2f}MB. Continuing to MP3 compression...")
                        current_path = flac_path
                        is_current_temp = True

        # 4. Convert to MP3 (last resort)
        current_size_mb = os.path.getsize(current_path) / (1024 * 1024)
        if current_size_mb > max_size_mb:
            logger.info("Strategy 3: Converting to MP3 (128k mono)...")

            mp3_path = manager.get_path_for(FileOperation.CONVERTED_MP3, "mp3")

            # Try PyAV first
            mp3_success = False
            if is_pyav_available():
                logger.debug("Using PyAV for MP3 conversion")
                mp3_success = convert_to_mp3_pyav(current_path, mp3_path)

            # Fallback to ffmpeg subprocess via existing function
            if not mp3_success:
                logger.debug("PyAV MP3 conversion failed, using ffmpeg fallback")
                mp3_temp = convert_to_mp3(current_path, bitrate="128k")
                if mp3_temp:
                    import shutil
                    shutil.move(str(mp3_temp), str(mp3_path))
                    mp3_success = True

            if mp3_success and mp3_path.exists():
                size_mb = os.path.getsize(mp3_path) / (1024 * 1024)
                manager.register(mp3_path, FileOperation.CONVERTED_MP3, current_path)

                if size_mb <= max_size_mb:
                    logger.success(f"MP3 conversion successful. New size: {size_mb:.2f}MB")
                    bytes_per_sec = _calculate_bytes_per_second(mp3_path)
                    return OptimizationResult(
                        path=mp3_path,
                        is_temporary=True,
                        size_mb=size_mb,
                        bytes_per_second=bytes_per_sec,
                        intermediate_manager=manager
                    )
                else:
                    logger.warning(f"MP3 file: {size_mb:.2f}MB. Will attempt chunking if supported.")
                    bytes_per_sec = _calculate_bytes_per_second(mp3_path)
                    return OptimizationResult(
                        path=mp3_path,
                        is_temporary=True,
                        size_mb=size_mb,
                        bytes_per_second=bytes_per_sec,
                        intermediate_manager=manager
                    )

        # If we get here, return the current file (might be original or last processed)
        bytes_per_sec = _calculate_bytes_per_second(current_path)
        final_size_mb = os.path.getsize(current_path) / (1024 * 1024)

        return OptimizationResult(
            path=current_path,
            is_temporary=is_current_temp,
            size_mb=final_size_mb,
            bytes_per_second=bytes_per_sec,
            intermediate_manager=manager if is_current_temp else None
        )

    except Exception as e:
        # Preserve intermediates on error if requested
        if preserve_on_error:
            manager.preserve_on_error()
        else:
            manager.cleanup()
        raise e


def _cleanup_intermediates(files: List[Path]):
    """Helper to clean up intermediate files. Deprecated - use IntermediateFileManager."""
    for f in files:
        if f.exists():
            try:
                os.unlink(f)
            except Exception as e:
                logger.warning(f"Failed to delete intermediate file {f}: {e}")
