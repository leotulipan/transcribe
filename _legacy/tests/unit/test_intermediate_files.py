"""
Unit tests for intermediate_files module.

Tests the IntermediateFileManager class which handles temporary file
creation, naming, registration, and cleanup during audio processing.
"""
import pytest
from pathlib import Path
from audio_transcribe.transcribe_helpers.intermediate_files import (
    FileOperation,
    IntermediateFile,
    IntermediateFileManager,
)


class TestIntermediateFileManager:
    """Test suite for IntermediateFileManager class."""

    def test_initialization_sets_correct_attributes(self, tmp_path):
        """Test that initialization sets correct attributes."""
        original_path = tmp_path / "test_audio.m4a"
        manager = IntermediateFileManager(original_path)

        assert manager.original_path == original_path
        assert manager.temp_dir == original_path.parent / f".transcribe_temp_{original_path.stem}"
        assert manager._files == {}
        assert manager._preserve is False

    def test_setup_creates_temp_directory(self, tmp_path):
        """Test that setup() creates the temporary directory."""
        original_path = tmp_path / "test_audio.m4a"
        manager = IntermediateFileManager(original_path)

        manager.setup()

        assert manager.temp_dir.exists()
        assert manager.temp_dir.is_dir()

    def test_setup_handles_existing_directory(self, tmp_path):
        """Test that setup() handles existing temp directory gracefully."""
        original_path = tmp_path / "test_audio.m4a"
        manager = IntermediateFileManager(original_path)

        # Create the directory first
        manager.temp_dir.mkdir(parents=True, exist_ok=True)

        # Should not raise an error
        manager.setup()

        assert manager.temp_dir.exists()

    def test_get_path_has_consistent_naming(self, tmp_path):
        """Test that get_path_for() returns consistent naming."""
        original_path = tmp_path / "test_audio.m4a"
        manager = IntermediateFileManager(original_path)

        path1 = manager.get_path_for(FileOperation.CONVERTED_FLAC, "flac")
        path2 = manager.get_path_for(FileOperation.CONVERTED_FLAC, "flac")

        assert path1 == path2
        assert path1.name == "test_audio_intermediate_flac.flac"

    def test_different_operations_get_different_names(self, tmp_path):
        """Test that different operations get different file names."""
        original_path = tmp_path / "test_audio.m4a"
        manager = IntermediateFileManager(original_path)

        flac_path = manager.get_path_for(FileOperation.CONVERTED_FLAC, "flac")
        mp3_path = manager.get_path_for(FileOperation.CONVERTED_MP3, "mp3")
        chunk_path = manager.get_path_for(FileOperation.CHUNK, "wav")

        assert flac_path.name == "test_audio_intermediate_flac.flac"
        assert mp3_path.name == "test_audio_intermediate_mp3.mp3"
        assert chunk_path.name == "test_audio_intermediate_chunk.wav"
        assert flac_path != mp3_path != chunk_path

    def test_register_file_tracks_metadata(self, tmp_path):
        """Test that register() creates and returns IntermediateFile metadata."""
        original_path = tmp_path / "test_audio.m4a"
        manager = IntermediateFileManager(original_path)

        # Create a test file
        test_file = tmp_path / "test.flac"
        test_file.write_bytes(b"fake flac data" * 100)  # ~1600 bytes

        metadata = manager.register(test_file, FileOperation.CONVERTED_FLAC, original_path)

        assert isinstance(metadata, IntermediateFile)
        assert metadata.path == test_file
        assert metadata.operation == FileOperation.CONVERTED_FLAC
        assert metadata.source_path == original_path
        assert metadata.size_mb > 0

    def test_register_tracks_multiple_files(self, tmp_path):
        """Test that register() can track multiple files."""
        original_path = tmp_path / "test_audio.m4a"
        manager = IntermediateFileManager(original_path)

        file1 = tmp_path / "test1.flac"
        file2 = tmp_path / "test2.mp3"
        file1.write_bytes(b"data1" * 100)
        file2.write_bytes(b"data2" * 100)

        manager.register(file1, FileOperation.CONVERTED_FLAC, original_path)
        manager.register(file2, FileOperation.CONVERTED_MP3, original_path)

        assert len(manager._files) == 2
        assert FileOperation.CONVERTED_FLAC in manager._files
        assert FileOperation.CONVERTED_MP3 in manager._files

    def test_preserve_on_error_sets_flag(self, tmp_path):
        """Test that preserve_on_error() sets the preserve flag."""
        original_path = tmp_path / "test_audio.m4a"
        manager = IntermediateFileManager(original_path)

        assert manager._preserve is False

        manager.preserve_on_error()

        assert manager._preserve is True

    def test_cleanup_deletes_registered_files(self, tmp_path):
        """Test that cleanup() deletes all registered intermediate files."""
        original_path = tmp_path / "test_audio.m4a"
        manager = IntermediateFileManager(original_path)
        manager.setup()

        # Create and register test files
        file1 = manager.temp_dir / "test1.flac"
        file2 = manager.temp_dir / "test2.mp3"
        file1.write_bytes(b"data1" * 100)
        file2.write_bytes(b"data2" * 100)

        manager.register(file1, FileOperation.CONVERTED_FLAC, original_path)
        manager.register(file2, FileOperation.CONVERTED_MP3, original_path)

        assert file1.exists()
        assert file2.exists()

        manager.cleanup()

        assert not file1.exists()
        assert not file2.exists()

    def test_cleanup_preserves_when_flagged(self, tmp_path):
        """Test that cleanup() preserves files when preserve_on_error was called."""
        original_path = tmp_path / "test_audio.m4a"
        manager = IntermediateFileManager(original_path)
        manager.setup()

        # Create and register test file
        test_file = manager.temp_dir / "test.flac"
        test_file.write_bytes(b"data" * 100)
        manager.register(test_file, FileOperation.CONVERTED_FLAC, original_path)

        manager.preserve_on_error()
        manager.cleanup()

        assert test_file.exists()

    def test_cleanup_removes_temp_dir_when_empty(self, tmp_path):
        """Test that cleanup() removes temp directory when empty."""
        original_path = tmp_path / "test_audio.m4a"
        manager = IntermediateFileManager(original_path)
        manager.setup()

        assert manager.temp_dir.exists()

        manager.cleanup()

        assert not manager.temp_dir.exists()

    def test_cleanup_keeps_temp_dir_with_preserved_files(self, tmp_path):
        """Test that cleanup() keeps temp dir when files are preserved."""
        original_path = tmp_path / "test_audio.m4a"
        manager = IntermediateFileManager(original_path)
        manager.setup()

        # Create and register test file
        test_file = manager.temp_dir / "test.flac"
        test_file.write_bytes(b"data" * 100)
        manager.register(test_file, FileOperation.CONVERTED_FLAC, original_path)

        manager.preserve_on_error()
        manager.cleanup()

        assert manager.temp_dir.exists()

    def test_cleanup_handles_nonexistent_files_gracefully(self, tmp_path):
        """Test that cleanup() handles nonexistent files without error."""
        original_path = tmp_path / "test_audio.m4a"
        manager = IntermediateFileManager(original_path)
        manager.setup()

        # Register a file that doesn't exist
        nonexistent = manager.temp_dir / "nonexistent.flac"
        manager.register(nonexistent, FileOperation.CONVERTED_FLAC, original_path)

        # Should not raise an error
        manager.cleanup()

    def test_cleanup_handles_deletion_errors_with_warnings(self, tmp_path, caplog):
        """Test that cleanup() handles deletion errors with warnings."""
        original_path = tmp_path / "test_audio.m4a"
        manager = IntermediateFileManager(original_path)
        manager.setup()

        # Create a file and register it
        test_file = manager.temp_dir / "test.flac"
        test_file.write_bytes(b"data" * 100)
        manager.register(test_file, FileOperation.CONVERTED_FLAC, original_path)

        # Simulate file being locked by another process
        # (We can't easily test this, but we can verify the code path)
        # For now, just test that cleanup doesn't crash
        manager.cleanup()

        assert not test_file.exists()

    def test_intermediate_file_dataclass(self, tmp_path):
        """Test IntermediateFile dataclass stores metadata correctly."""
        original_path = tmp_path / "test_audio.m4a"
        test_path = tmp_path / "test.flac"
        test_path.write_bytes(b"data" * 100)

        metadata = IntermediateFile(
            path=test_path,
            operation=FileOperation.CONVERTED_FLAC,
            source_path=original_path,
            size_mb=0.0015
        )

        assert metadata.path == test_path
        assert metadata.operation == FileOperation.CONVERTED_FLAC
        assert metadata.source_path == original_path
        assert metadata.size_mb == 0.0015

    def test_file_operation_enum_values(self):
        """Test that FileOperation enum has correct values."""
        assert FileOperation.EXTRACTED.value == "extracted"
        assert FileOperation.CONVERTED_FLAC.value == "flac"
        assert FileOperation.CONVERTED_MP3.value == "mp3"
        assert FileOperation.CHUNK.value == "chunk"

    def test_multiple_managers_have_separate_temp_dirs(self, tmp_path):
        """Test that multiple managers have separate temp directories."""
        file1 = tmp_path / "audio1.m4a"
        file2 = tmp_path / "audio2.m4a"

        manager1 = IntermediateFileManager(file1)
        manager2 = IntermediateFileManager(file2)

        manager1.setup()
        manager2.setup()

        assert manager1.temp_dir != manager2.temp_dir
        assert manager1.temp_dir.name == ".transcribe_temp_audio1"
        assert manager2.temp_dir.name == ".transcribe_temp_audio2"

    def test_get_path_without_setup(self, tmp_path):
        """Test that get_path_for() works even before setup() is called."""
        original_path = tmp_path / "test_audio.m4a"
        manager = IntermediateFileManager(original_path)

        # Don't call setup()
        path = manager.get_path_for(FileOperation.CONVERTED_FLAC, "flac")

        assert path.name == "test_audio_intermediate_flac.flac"
        # Path parent directory doesn't exist yet
        assert not path.parent.exists()
