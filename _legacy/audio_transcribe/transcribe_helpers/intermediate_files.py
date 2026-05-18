"""
Intermediate file management for audio processing.

Provides consistent naming and deferred cleanup for intermediate files
created during audio optimization (extraction, conversion, chunking).
"""
import hashlib
import sys
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional
from loguru import logger

# Windows MAX_PATH is 260; reserve space for intermediate file names
_MAX_PATH = 260
# Longest intermediate filename: {stem}_intermediate_extracted.flac = stem + 27 chars
_INTERMEDIATE_SUFFIX_MAX = 30
# Prefix for temp dir: .transcribe_temp_ = 17 chars
_TEMP_DIR_PREFIX = ".transcribe_temp_"


class FileOperation(Enum):
    """Types of intermediate file operations."""
    EXTRACTED = "extracted"
    CONVERTED_FLAC = "flac"
    CONVERTED_MP3 = "mp3"
    CHUNK = "chunk"


@dataclass
class IntermediateFile:
    """Metadata for an intermediate file."""
    path: Path
    operation: FileOperation
    source_path: Path
    size_mb: float = 0.0


class IntermediateFileManager:
    """
    Manages intermediate files with consistent naming and deferred cleanup.

    Features:
    - Consistent naming: {stem}_intermediate_{operation}.{ext}
    - Temp directory alongside original file
    - Deferred cleanup (only after success)
    - Error preservation (on failure, temp dir is kept)

    Usage:
        manager = IntermediateFileManager(input_path)
        manager.setup()
        flac_path = manager.get_path_for(FileOperation.CONVERTED_FLAC, "flac")
        # ... process file ...
        if success:
            manager.cleanup()
        else:
            manager.preserve_on_error()
    """

    def __init__(self, original_path: Path):
        """
        Initialize the manager.

        Args:
            original_path: Path to the original input file
        """
        self.original_path = Path(original_path)
        self._short_stem = self._make_short_stem()
        self.temp_dir = self.original_path.parent / f"{_TEMP_DIR_PREFIX}{self._short_stem}"
        self._files: Dict[FileOperation, IntermediateFile] = {}
        self._preserve = False

    def _make_short_stem(self) -> str:
        """Return a stem short enough to keep full intermediate paths under MAX_PATH on Windows."""
        stem = self.original_path.stem
        if sys.platform != "win32":
            return stem

        # Budget: MAX_PATH - parent_dir - separators - temp_dir_prefix - intermediate_suffix
        parent_len = len(str(self.original_path.parent))
        # temp_dir path = parent / .transcribe_temp_{stem} / {stem}_intermediate_xxx.ext
        # total = parent + 1 + prefix + stem + 1 + stem + _intermediate_suffix
        # so stem appears twice, plus fixed overhead
        overhead = parent_len + 1 + len(_TEMP_DIR_PREFIX) + 1 + _INTERMEDIATE_SUFFIX_MAX
        available = _MAX_PATH - overhead
        # stem appears twice (dir name + file name)
        max_stem = available // 2

        if max_stem < 8:
            # Extremely long parent path; use just a hash
            return hashlib.md5(stem.encode()).hexdigest()[:16]

        if len(stem) <= max_stem:
            return stem

        # Truncate and append short hash for uniqueness
        hash_suffix = hashlib.md5(stem.encode()).hexdigest()[:8]
        truncated = stem[:max_stem - 9]  # 9 = 1 underscore + 8 hash chars
        return f"{truncated}_{hash_suffix}"

    def setup(self) -> None:
        """Create the temporary directory for intermediate files."""
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def get_path_for(self, operation: FileOperation, ext: str) -> Path:
        """
        Get a consistent path for an intermediate file.

        Args:
            operation: Type of file operation
            ext: File extension (e.g., "flac", "mp3")

        Returns:
            Path for the intermediate file
        """
        return self.temp_dir / f"{self._short_stem}_intermediate_{operation.value}.{ext}"

    def register(self, path: Path, operation: FileOperation, source: Path) -> IntermediateFile:
        """
        Register an intermediate file for tracking.

        Args:
            path: Path to the intermediate file
            operation: Type of operation that created the file
            source: Source file that was processed

        Returns:
            IntermediateFile metadata
        """
        size_mb = path.stat().st_size / (1024 * 1024) if path.exists() else 0.0
        f = IntermediateFile(path, operation, source, size_mb)
        self._files[operation] = f
        return f

    def preserve_on_error(self) -> None:
        """Mark intermediate files to be preserved (don't cleanup)."""
        self._preserve = True
        logger.info(f"Preserving intermediates in: {self.temp_dir}")

    # Files that can be safely deleted during cleanup (OS/cloud-generated)
    _IGNORABLE_FILES = {"desktop.ini", "thumbs.db", ".ds_store"}

    def cleanup(self) -> None:
        """Clean up all registered intermediate files and temp directory."""
        if self._preserve:
            logger.debug("Skipping cleanup (preserve_on_error was called)")
            return

        for f in self._files.values():
            if f.path.exists():
                try:
                    f.path.unlink()
                    logger.debug(f"Deleted intermediate file: {f.path}")
                except Exception as e:
                    logger.warning(f"Failed to delete intermediate file {f.path}: {e}")

        # Remove temp directory, cleaning up OS-generated files first
        if self.temp_dir.exists():
            remaining = list(self.temp_dir.iterdir())
            # Delete ignorable OS/cloud-generated files (desktop.ini, Thumbs.db, etc.)
            for item in remaining:
                if item.name.lower() in self._IGNORABLE_FILES:
                    try:
                        item.unlink()
                        logger.debug(f"Deleted OS-generated file: {item}")
                    except Exception as e:
                        logger.debug(f"Could not delete {item}: {e}")

            # Now try to remove the directory
            if not any(self.temp_dir.iterdir()):
                try:
                    self.temp_dir.rmdir()
                    logger.debug(f"Removed temporary directory: {self.temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to remove temporary directory {self.temp_dir}: {e}")
