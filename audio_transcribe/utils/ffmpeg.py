"""FFmpeg detection and configuration utilities."""

import os
import shutil
from pathlib import Path
from typing import Optional

from loguru import logger

# Cache for ffmpeg/ffprobe paths (avoid repeated filesystem lookups)
_ffmpeg_cache: Optional[str] = None
_ffprobe_cache: Optional[str] = None
_cache_initialized: bool = False


def find_ffmpeg() -> Optional[str]:
    """Find FFmpeg executable using multiple strategies.

    Search order:
    1. System PATH (most reliable, user has it installed properly)
    2. FFMPEG_PATH environment variable (explicit user override)
    3. Common Windows install locations
    4. Local bundled copy (for portable/standalone builds)

    Returns:
        Path to ffmpeg executable, or None if not found.
    """
    global _ffmpeg_cache, _cache_initialized

    # Return cached result if available
    if _cache_initialized and _ffmpeg_cache is not None:
        return _ffmpeg_cache

    # 1. Check PATH first (most reliable - user installed it properly)
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        logger.debug(f"Found ffmpeg in PATH: {ffmpeg_path}")
        _ffmpeg_cache = ffmpeg_path
        _cache_initialized = True
        return ffmpeg_path

    # 2. Check environment variable (explicit user override)
    env_path = os.environ.get("FFMPEG_PATH")
    if env_path:
        env_path_obj = Path(env_path)
        # Handle both directory and direct executable path
        if env_path_obj.is_file():
            logger.debug(f"Found ffmpeg via FFMPEG_PATH: {env_path}")
            _ffmpeg_cache = str(env_path_obj)
            _cache_initialized = True
            return _ffmpeg_cache
        elif env_path_obj.is_dir():
            ffmpeg_exe = env_path_obj / "ffmpeg.exe"
            if ffmpeg_exe.exists():
                logger.debug(f"Found ffmpeg via FFMPEG_PATH dir: {ffmpeg_exe}")
                _ffmpeg_cache = str(ffmpeg_exe)
                _cache_initialized = True
                return _ffmpeg_cache

    # 3. Check common Windows install locations
    common_paths = [
        # WinGet install location (most common on modern Windows)
        Path(os.path.expanduser(r"~\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe")),
        Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
        Path(r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"),
        Path(r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe"),
        Path(os.path.expanduser(r"~\ffmpeg\bin\ffmpeg.exe")),
        Path(os.path.expanduser(r"~\scoop\apps\ffmpeg\current\bin\ffmpeg.exe")),
        Path(r"C:\ProgramData\chocolatey\bin\ffmpeg.exe"),
    ]

    for path in common_paths:
        if path.exists():
            logger.debug(f"Found ffmpeg at common location: {path}")
            _ffmpeg_cache = str(path)
            _cache_initialized = True
            return _ffmpeg_cache

    # 4. Check for local bundled copy (portable/standalone builds)
    local_paths = [
        Path.cwd() / "ffmpeg" / "bin" / "ffmpeg.exe",
        Path.cwd() / "ffmpeg" / "ffmpeg.exe",
        Path.cwd() / "ffmpeg.exe",
    ]

    for path in local_paths:
        if path.exists():
            logger.debug(f"Found bundled ffmpeg: {path}")
            _ffmpeg_cache = str(path)
            return _ffmpeg_cache

    # Cache the "not found" state
    _cache_initialized = True
    return None


def find_ffprobe() -> Optional[str]:
    """Find FFprobe executable using the same strategies as ffmpeg."""
    global _ffprobe_cache

    # Return cached result if available
    if _ffprobe_cache is not None:
        return _ffprobe_cache

    # 1. Check PATH
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path:
        _ffprobe_cache = ffprobe_path
        return ffprobe_path

    # 2. If we found ffmpeg, ffprobe is usually next to it
    ffmpeg_path = find_ffmpeg()
    if ffmpeg_path:
        ffmpeg_dir = Path(ffmpeg_path).parent
        ffprobe_exe = ffmpeg_dir / "ffprobe.exe"
        if ffprobe_exe.exists():
            _ffprobe_cache = str(ffprobe_exe)
            return _ffprobe_cache
        # Unix variant
        ffprobe_unix = ffmpeg_dir / "ffprobe"
        if ffprobe_unix.exists():
            _ffprobe_cache = str(ffprobe_unix)
            return _ffprobe_cache

    return None


def configure_pydub() -> bool:
    """Configure pydub to use detected ffmpeg/ffprobe paths.

    This also adds the ffmpeg directory to PATH so that pydub's internal
    functions (which use their own `which()` lookups) can find ffprobe.

    Returns:
        True if ffmpeg was found and configured, False otherwise.
    """
    from pydub import AudioSegment

    ffmpeg_path = find_ffmpeg()
    ffprobe_path = find_ffprobe()

    if ffmpeg_path:
        AudioSegment.converter = ffmpeg_path
        logger.debug(f"Configured pydub converter: {ffmpeg_path}")

        # Add ffmpeg directory to PATH for pydub's internal which() calls
        ffmpeg_dir = str(Path(ffmpeg_path).parent)
        current_path = os.environ.get("PATH", "")
        if ffmpeg_dir not in current_path:
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + current_path
            logger.debug(f"Added {ffmpeg_dir} to PATH")

    if ffprobe_path:
        AudioSegment.ffprobe = ffprobe_path
        logger.debug(f"Configured pydub ffprobe: {ffprobe_path}")

    return ffmpeg_path is not None


def require_ffmpeg() -> str:
    """Get ffmpeg path or raise an error with helpful instructions.

    Returns:
        Path to ffmpeg executable.

    Raises:
        RuntimeError: If ffmpeg is not found.
    """
    ffmpeg_path = find_ffmpeg()
    if not ffmpeg_path:
        raise RuntimeError(
            "FFmpeg not found. Please install FFmpeg:\n"
            "  - Windows: winget install ffmpeg  OR  choco install ffmpeg  OR  scoop install ffmpeg\n"
            "  - Or download from https://ffmpeg.org/download.html and add to PATH\n"
            "  - Or set FFMPEG_PATH environment variable to the ffmpeg.exe location"
        )
    return ffmpeg_path
