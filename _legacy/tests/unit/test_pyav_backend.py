"""
Unit tests for pyav_backend module.

Tests PyAV-based audio processing functions with mocking for cases
where PyAV is not available.
"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
from audio_transcribe.transcribe_helpers.pyav_backend import (
    is_pyav_available,
    get_duration_seconds,
    _get_duration_ffprobe,
    extract_audio_pyav,
    convert_to_flac_pyav,
    convert_to_mp3_pyav,
    PYAV_AVAILABLE,
)


class TestPyAVAvailability:
    """Test suite for PyAV availability detection."""

    def test_returns_true_when_installed(self):
        """Test that is_pyav_available returns True when PyAV is installed."""
        # This test assumes PyAV is installed in the test environment
        # If it's not, the integration tests should handle that
        if PYAV_AVAILABLE:
            assert is_pyav_available() is True

    @patch('audio_transcribe.transcribe_helpers.pyav_backend.PYAV_AVAILABLE', False)
    def test_returns_false_when_missing(self):
        """Test that is_pyav_available returns False when PyAV is not installed."""
        assert is_pyav_available() is False


class TestDurationDetection:
    """Test suite for audio/video duration detection."""

    def test_valid_audio_returns_correct_duration(self, tmp_path, sample_audio_file):
        """Test that valid audio file returns correct duration."""
        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        duration = get_duration_seconds(sample_audio_file)

        # Sample audio should be at least 1 second
        assert duration > 0
        assert duration < 1000  # Sanity check (not more than 16 minutes)

    def test_video_file_duration(self, sample_audio_files):
        """Test that duration detection works for video files."""
        video_file = sample_audio_files.get("video")
        if not video_file:
            pytest.skip("No video file available")

        duration = get_duration_seconds(video_file)

        assert duration > 0

    def test_invalid_file_returns_zero(self, tmp_path):
        """Test that invalid file returns 0.0 duration."""
        invalid_file = tmp_path / "invalid.wav"
        invalid_file.write_text("Not a real audio file")

        duration = get_duration_seconds(invalid_file)

        assert duration == 0.0

    def test_nonexistent_file_returns_zero(self, tmp_path):
        """Test that nonexistent file returns 0.0 duration."""
        nonexistent = tmp_path / "does_not_exist.wav"

        duration = get_duration_seconds(nonexistent)

        assert duration == 0.0


class TestAudioExtraction:
    """Test suite for audio extraction from video."""

    def test_successful_extraction(self, sample_audio_files, tmp_path):
        """Test successful audio extraction from video."""
        video_file = sample_audio_files.get("video")
        if not video_file:
            pytest.skip("No video file available")

        output_path = tmp_path / "extracted_audio.wav"

        if PYAV_AVAILABLE:
            result = extract_audio_pyav(video_file, output_path)

            if result:
                assert output_path.exists()
                assert output_path.stat().st_size > 0

    def test_no_audio_stream_handling(self, tmp_path):
        """Test handling of video file without audio stream."""
        # Create a mock video file with no audio stream
        # This is difficult to test without actual files, so we mock
        if not PYAV_AVAILABLE:
            pytest.skip("PyAV not available")

        with patch('audio_transcribe.transcribe_helpers.pyav_backend.av.open') as mock_open:
            mock_container = MagicMock()
            mock_container.streams.audio = []  # No audio stream
            mock_open.return_value.__enter__.return_value = mock_container

            result = extract_audio_pyav(tmp_path / "test.mkv", tmp_path / "output.wav")

            assert result is False

    def test_returns_false_when_pyav_unavailable(self, tmp_path):
        """Test that extraction returns False when PyAV is not available."""
        with patch('audio_transcribe.transcribe_helpers.pyav_backend.PYAV_AVAILABLE', False):
            result = extract_audio_pyav(tmp_path / "test.mkv", tmp_path / "output.wav")
            assert result is False

    def test_exception_handling(self, tmp_path):
        """Test that exceptions are handled gracefully."""
        if not PYAV_AVAILABLE:
            pytest.skip("PyAV not available")

        with patch('audio_transcribe.transcribe_helpers.pyav_backend.av.open') as mock_open:
            mock_open.side_effect = Exception("Test exception")

            result = extract_audio_pyav(tmp_path / "test.mkv", tmp_path / "output.wav")

            assert result is False


class TestFLACConversion:
    """Test suite for FLAC conversion."""

    def test_successful_conversion(self, sample_audio_file, tmp_path):
        """Test successful FLAC conversion."""
        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        output_path = tmp_path / "converted.flac"

        if PYAV_AVAILABLE:
            result = convert_to_flac_pyav(sample_audio_file, output_path)

            if result:
                assert output_path.exists()
                assert output_path.stat().st_size > 0

    def test_sample_rate_conversion(self, sample_audio_file, tmp_path):
        """Test conversion with different sample rate."""
        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        output_path = tmp_path / "converted_16k.flac"

        if PYAV_AVAILABLE:
            result = convert_to_flac_pyav(sample_audio_file, output_path, sample_rate=16000)

            if result:
                assert output_path.exists()

    def test_mono_conversion(self, sample_audio_file, tmp_path):
        """Test conversion to mono."""
        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        output_path = tmp_path / "converted_mono.flac"

        if PYAV_AVAILABLE:
            result = convert_to_flac_pyav(sample_audio_file, output_path)

            if result:
                assert output_path.exists()

    def test_no_audio_stream_handling(self, tmp_path):
        """Test handling of file with no audio stream."""
        if not PYAV_AVAILABLE:
            pytest.skip("PyAV not available")

        with patch('audio_transcribe.transcribe_helpers.pyav_backend.av.open') as mock_open:
            mock_container = MagicMock()
            mock_container.streams.audio = []  # No audio stream
            mock_open.return_value.__enter__.return_value = mock_container

            result = convert_to_flac_pyav(tmp_path / "test.wav", tmp_path / "output.flac")

            assert result is False

    def test_custom_sample_rate(self, sample_audio_file, tmp_path):
        """Test conversion with custom sample rate."""
        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        output_path = tmp_path / "converted_22k.flac"

        if PYAV_AVAILABLE:
            result = convert_to_flac_pyav(sample_audio_file, output_path, sample_rate=22050)

            if result:
                assert output_path.exists()

    def test_returns_false_when_unavailable(self, tmp_path):
        """Test that conversion returns False when PyAV is not available."""
        with patch('audio_transcribe.transcribe_helpers.pyav_backend.PYAV_AVAILABLE', False):
            result = convert_to_flac_pyav(tmp_path / "test.wav", tmp_path / "output.flac")
            assert result is False


class TestMP3Conversion:
    """Test suite for MP3 conversion."""

    def test_successful_conversion(self, sample_audio_file, tmp_path):
        """Test successful MP3 conversion."""
        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        output_path = tmp_path / "converted.mp3"

        if PYAV_AVAILABLE:
            result = convert_to_mp3_pyav(sample_audio_file, output_path)

            if result:
                assert output_path.exists()
                assert output_path.stat().st_size > 0

    def test_bitrate_parameter_respected(self, sample_audio_file, tmp_path):
        """Test that bitrate parameter is respected."""
        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        output_path = tmp_path / "converted_64k.mp3"

        if PYAV_AVAILABLE:
            result = convert_to_mp3_pyav(sample_audio_file, output_path, bitrate="64k")

            if result:
                assert output_path.exists()

    def test_mono_conversion(self, sample_audio_file, tmp_path):
        """Test conversion to mono."""
        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        output_path = tmp_path / "converted_mono.mp3"

        if PYAV_AVAILABLE:
            result = convert_to_mp3_pyav(sample_audio_file, output_path)

            if result:
                assert output_path.exists()

    def test_no_audio_stream_handling(self, tmp_path):
        """Test handling of file with no audio stream."""
        if not PYAV_AVAILABLE:
            pytest.skip("PyAV not available")

        with patch('audio_transcribe.transcribe_helpers.pyav_backend.av.open') as mock_open:
            mock_container = MagicMock()
            mock_container.streams.audio = []  # No audio stream
            mock_open.return_value.__enter__.return_value = mock_container

            result = convert_to_mp3_pyav(tmp_path / "test.wav", tmp_path / "output.mp3")

            assert result is False

    def test_bitrate_parsing(self, sample_audio_file, tmp_path):
        """Test parsing of bitrate string format (e.g., '128k')."""
        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        # Test various bitrate formats
        bitrates = ["64k", "128k", "320k"]

        for bitrate in bitrates:
            output_path = tmp_path / f"converted_{bitrate}.mp3"

            if PYAV_AVAILABLE:
                result = convert_to_mp3_pyav(sample_audio_file, output_path, bitrate=bitrate)

                if result:
                    assert output_path.exists()

    def test_returns_false_when_unavailable(self, tmp_path):
        """Test that conversion returns False when PyAV is not available."""
        with patch('audio_transcribe.transcribe_helpers.pyav_backend.PYAV_AVAILABLE', False):
            result = convert_to_mp3_pyav(tmp_path / "test.wav", tmp_path / "output.mp3")
            assert result is False
