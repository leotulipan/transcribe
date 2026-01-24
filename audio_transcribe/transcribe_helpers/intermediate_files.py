"""
Intermediate file management for audio processing.

Provides consistent naming and deferred cleanup for intermediate files
created during audio optimization (extraction, conversion, chunking).
"""
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional
from loguru import logger


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
        self.temp_dir = self.original_path.parent / f".transcribe_temp_{self.original_path.stem}"
        self._files: Dict[FileOperation, IntermediateFile] = {}
        self._preserve = False

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
        return self.temp_dir / f"{self.original_path.stem}_intermediate_{operation.value}.{ext}"

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

        # Remove temp directory if empty
        if self.temp_dir.exists() and not any(self.temp_dir.iterdir()):
            try:
                self.temp_dir.rmdir()
                logger.debug(f"Removed temporary directory: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary directory {self.temp_dir}: {e}")
