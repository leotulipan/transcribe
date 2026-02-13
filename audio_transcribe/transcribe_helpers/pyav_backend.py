"""
PyAV backend for fast audio processing.

Provides PyAV-based implementations with ffmpeg subprocess fallbacks
for audio extraction, conversion, and duration detection.
"""
import subprocess
from pathlib import Path
from typing import Optional
from loguru import logger

PYAV_AVAILABLE = False
try:
    import av
    PYAV_AVAILABLE = True
except ImportError:
    pass


def is_pyav_available() -> bool:
    """Check if PyAV is available."""
    return PYAV_AVAILABLE


def get_duration_seconds(audio_path: Path) -> float:
    """
    Get audio/video duration in seconds without subprocess (10x faster than ffprobe).

    Args:
        audio_path: Path to audio/video file

    Returns:
        Duration in seconds, or 0.0 if unable to determine
    """
    if PYAV_AVAILABLE:
        try:
            with av.open(str(audio_path)) as container:
                if container.duration:
                    # duration is in microseconds
                    return container.duration / 1_000_000
                # Fallback: calculate from streams
                if container.streams.audio:
                    stream = container.streams.audio[0]
                    # For some formats, duration may be None
                    # Try to get from context
                    if hasattr(stream, 'duration') and stream.duration:
                        return stream.duration / 1_000_000
        except Exception as e:
            logger.debug(f"PyAV duration detection failed: {e}")

    # Fallback to ffprobe
    return _get_duration_ffprobe(audio_path)


def _get_duration_ffprobe(audio_path: Path) -> float:
    """
    Get duration using ffprobe subprocess.

    Args:
        audio_path: Path to audio/video file

    Returns:
        Duration in seconds, or 0.0 if unable to determine
    """
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', str(audio_path)],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass
    return 0.0


def extract_audio_pyav(input_path: Path, output_path: Path) -> bool:
    """
    Stream copy audio from video (2-3x faster than subprocess).

    Args:
        input_path: Path to input video file
        output_path: Path to output audio file

    Returns:
        True if successful, False otherwise
    """
    if not PYAV_AVAILABLE:
        return False

    try:
        with av.open(str(input_path)) as inp:
            # Find audio stream
            audio_streams = inp.streams.audio
            if not audio_streams:
                logger.warning("No audio stream found in input file")
                return False

            audio_stream = audio_streams[0]
            codec_name = audio_stream.codec_context.name

            with av.open(str(output_path), 'w') as out:
                # Add stream matching input codec for stream copy
                out_stream = out.add_stream(codec_name, rate=audio_stream.rate)
                out_stream.layout = audio_stream.layout

                # mux packets
                for packet in inp.demux(audio_stream):
                    if packet.dts is None:
                        continue
                    packet.stream = out_stream
                    out.mux(packet)

        return True
    except Exception as e:
        logger.warning(f"PyAV extraction failed: {e}")
        return False


def convert_to_flac_pyav(input_path: Path, output_path: Path, sample_rate: int = 16000) -> bool:
    """
    Convert to FLAC 16kHz mono using PyAV.

    Args:
        input_path: Path to input audio/video file
        output_path: Path to output FLAC file
        sample_rate: Target sample rate (default 16000)

    Returns:
        True if successful, False otherwise
    """
    if not PYAV_AVAILABLE:
        return False

    try:
        with av.open(str(input_path)) as inp:
            # Find audio stream
            audio_streams = inp.streams.audio
            if not audio_streams:
                logger.warning("No audio stream found in input file")
                return False

            audio_stream = audio_streams[0]

            with av.open(str(output_path), 'w') as out:
                # Create FLAC codec with 16kHz mono
                out_stream = out.add_stream('flac', rate=sample_rate)
                out_stream.layout = 'mono'

                # Setup resampler if needed
                resampler = None
                if audio_stream.sample_rate != sample_rate or audio_stream.layout != 'mono':
                    resampler = av.AudioResampler(
                        format='s16',
                        layout='mono',
                        rate=sample_rate
                    )

                # Decode and encode frames
                for frame in inp.decode(audio_stream):
                    if resampler:
                        for resampled in resampler.resample(frame):
                            for packet in out_stream.encode(resampled):
                                out.mux(packet)
                    else:
                        for packet in out_stream.encode(frame):
                            out.mux(packet)

                # Flush encoder
                for packet in out_stream.encode():
                    out.mux(packet)

        return True
    except Exception as e:
        logger.warning(f"PyAV FLAC conversion failed: {e}")
        return False


def convert_to_mp3_pyav(input_path: Path, output_path: Path, bitrate: str = "128k") -> bool:
    """
    Convert to MP3 using PyAV.

    Args:
        input_path: Path to input audio/video file
        output_path: Path to output MP3 file
        bitrate: Target bitrate (e.g., "128k", "64k")

    Returns:
        True if successful, False otherwise
    """
    if not PYAV_AVAILABLE:
        return False

    try:
        with av.open(str(input_path)) as inp:
            # Find audio stream
            audio_streams = inp.streams.audio
            if not audio_streams:
                logger.warning("No audio stream found in input file")
                return False

            audio_stream = audio_streams[0]

            with av.open(str(output_path), 'w') as out:
                # Create MP3 codec with specified bitrate
                # Extract numeric part from bitrate (e.g., "128k" -> 128000)
                bitrate_num = int(bitrate.rstrip('k')) * 1000
                out_stream = out.add_stream('mp3', rate=audio_stream.sample_rate)
                out_stream.bit_rate = bitrate_num
                out_stream.layout = 'mono'

                # Setup resampler if needed
                resampler = None
                if audio_stream.layout != 'mono':
                    resampler = av.AudioResampler(
                        format='s16',
                        layout='mono',
                        rate=audio_stream.sample_rate
                    )

                # Decode and encode frames
                for frame in inp.decode(audio_stream):
                    if resampler:
                        for resampled in resampler.resample(frame):
                            for packet in out_stream.encode(resampled):
                                out.mux(packet)
                    else:
                        for packet in out_stream.encode(frame):
                            out.mux(packet)

                # Flush encoder
                for packet in out_stream.encode():
                    out.mux(packet)

        return True
    except Exception as e:
        logger.warning(f"PyAV MP3 conversion failed: {e}")
        return False
