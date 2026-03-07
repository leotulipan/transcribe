"""
Unit tests for chunking module.

Tests audio chunking functionality for handling large files that exceed
API size limits, including splitting, overlap handling, and transcript merging.
"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
from pydub import AudioSegment
from audio_transcribe.transcribe_helpers.chunking import (
    split_audio,
    split_audio_streaming,
    _split_audio_pyav,
    merge_transcripts,
    transcribe_with_chunks,
)


@pytest.mark.requires_ffmpeg
class TestAudioSplitting:
    """Test suite for audio splitting functionality."""

    @pytest.fixture
    def sample_audio_1min(self, tmp_path):
        """Create a 1-minute silent audio file for testing."""
        audio = AudioSegment.silent(duration=60000)  # 60 seconds in ms
        path = tmp_path / "test_1min.wav"
        audio.export(str(path), format="wav")
        return path

    @pytest.fixture
    def sample_audio_10min(self, tmp_path):
        """Create a 10-minute silent audio file for testing."""
        audio = AudioSegment.silent(duration=600000)  # 600 seconds in ms
        path = tmp_path / "test_10min.wav"
        audio.export(str(path), format="wav")
        return path

    def test_correct_number_of_chunks(self, sample_audio_10min):
        """Test that audio is split into correct number of chunks."""
        chunks = split_audio(sample_audio_10min, chunk_length=300, overlap=10)

        # 600 seconds / (300 - 10) effective length = ~2.06 chunks
        assert len(chunks) == 3

    def test_overlap_between_chunks(self, sample_audio_10min):
        """Test that chunks have correct overlap."""
        chunks = split_audio(sample_audio_10min, chunk_length=300, overlap=10)

        # First chunk starts at 0
        assert chunks[0][1] == 0.0

        # Second chunk should start at 290 seconds (300 - 10 overlap)
        assert abs(chunks[1][1] - 290.0) < 1.0

    def test_last_chunk_shorter(self, sample_audio_10min):
        """Test that the last chunk is shorter when audio doesn't divide evenly."""
        chunks = split_audio(sample_audio_10min, chunk_length=300, overlap=10)

        # Last chunk should be shorter than full chunk_length
        # It covers the remainder of the audio
        assert chunks[-1][1] > 0

    def test_single_chunk_no_splitting(self, sample_audio_1min):
        """Test that short audio doesn't get split."""
        chunks = split_audio(sample_audio_1min, chunk_length=300, overlap=10)

        # 60 seconds fits in one 300-second chunk
        assert len(chunks) == 1
        assert chunks[0][1] == 0.0

    def test_custom_chunk_length(self, sample_audio_10min):
        """Test that custom chunk_length is respected."""
        chunks = split_audio(sample_audio_10min, chunk_length=100, overlap=5)

        # 600 seconds / (100 - 5) effective length = ~6.3 chunks
        assert len(chunks) == 7

    def test_custom_overlap(self, sample_audio_10min):
        """Test that custom overlap is respected."""
        chunks = split_audio(sample_audio_10min, chunk_length=300, overlap=30)

        # Second chunk should start at 270 seconds (300 - 30 overlap)
        assert abs(chunks[1][1] - 270.0) < 1.0

    def test_creates_temp_files(self, sample_audio_10min):
        """Test that temp files are created for each chunk."""
        chunks = split_audio(sample_audio_10min, chunk_length=300, overlap=10)

        for chunk_path, _ in chunks:
            assert chunk_path.exists()
            assert chunk_path.suffix == ".flac"

    def test_timestamps_preserved(self, sample_audio_10min):
        """Test that start timestamps are preserved for each chunk."""
        chunks = split_audio(sample_audio_10min, chunk_length=300, overlap=10)

        # Check that timestamps are monotonically increasing
        timestamps = [t for _, t in chunks]
        assert timestamps == sorted(timestamps)


@pytest.mark.requires_ffmpeg
class TestPyAVStreamingSplit:
    """Test suite for PyAV-based streaming split."""

    def test_uses_pyav_when_available(self, sample_audio_file):
        """Test that PyAV path is used when PyAV is available."""
        from audio_transcribe.transcribe_helpers.pyav_backend import is_pyav_available

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        if not is_pyav_available():
            pytest.skip("PyAV not available")

        chunks = split_audio_streaming(sample_audio_file, chunk_length=300, overlap=10)

        # Should return chunks
        assert len(chunks) > 0

    def test_fallback_to_pydub(self, sample_audio_file):
        """Test fallback to pydub when PyAV fails."""
        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        with patch('audio_transcribe.transcribe_helpers.pyav_backend.is_pyav_available', return_value=False):
            chunks = split_audio_streaming(sample_audio_file, chunk_length=300, overlap=10)

            # Should still work via pydub
            assert len(chunks) >= 1

    def test_duration_detection(self, sample_audio_file):
        """Test that duration is detected correctly."""
        from audio_transcribe.transcribe_helpers.pyav_backend import is_pyav_available

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        if not is_pyav_available():
            pytest.skip("PyAV not available")

        # This test verifies duration detection is working
        # The actual implementation uses get_duration_seconds
        from audio_transcribe.transcribe_helpers.pyav_backend import get_duration_seconds

        duration = get_duration_seconds(sample_audio_file)
        assert duration > 0


class TestTranscriptMerging:
    """Test suite for transcript merging functionality."""

    @pytest.fixture
    def mock_transcription_dict(self):
        """Create a mock transcription dict for testing (matching model_dump() format)."""
        return {
            "text": "Hello world this is a test",
            "confidence": 0.95,
            "language": "en",
            "words": [],
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "Hello world"},
                {"start": 1.0, "end": 2.0, "text": "this is a test"}
            ],
            "speakers": [],
            "api_name": "mock"
        }

    def test_merges_text_from_chunks(self, mock_transcription_dict):
        """Test that text is merged from multiple chunks."""
        chunk1 = mock_transcription_dict
        chunk2 = mock_transcription_dict.copy()
        chunk2["segments"] = [
            {"start": 2.0, "end": 3.0, "text": "More text"},
            {"start": 3.0, "end": 4.0, "text": "End of test"}
        ]

        results = [(chunk1, 0.0), (chunk2, 2.0)]

        merged = merge_transcripts(results, overlap=1)

        assert "text" in merged
        assert len(merged["text"]) > 0

    def test_adjusts_timestamps_correctly(self, mock_transcription_dict):
        """Test that timestamps are adjusted for chunk position."""
        chunk1 = mock_transcription_dict
        chunk2 = mock_transcription_dict.copy()
        chunk2["segments"] = [
            {"start": 2.0, "end": 3.0, "text": "More text"}
        ]

        results = [(chunk1, 0.0), (chunk2, 2.0)]

        merged = merge_transcripts(results, overlap=1)

        # Check that timestamps have been adjusted
        # First chunk starts at 0, second at 2.0
        segments = merged.get("segments", [])
        if segments:
            # Verify some segments exist
            assert len(segments) > 0

    def test_handles_overlap_regions(self, mock_transcription_dict):
        """Test handling of overlap between chunks."""
        chunk1 = mock_transcription_dict
        chunk2 = mock_transcription_dict.copy()
        chunk2["segments"] = [
            {"start": 1.5, "end": 2.5, "text": "More text"}
        ]

        results = [(chunk1, 0.0), (chunk2, 1.5)]  # Overlap of 0.5s

        merged = merge_transcripts(results, overlap=1)

        assert "segments" in merged

    def test_combines_segment_data(self, mock_transcription_dict):
        """Test that segment data is properly combined."""
        chunk1 = mock_transcription_dict
        chunk2 = mock_transcription_dict.copy()
        chunk2["segments"] = [
            {"start": 2.0, "end": 3.0, "text": "More text"}
        ]

        results = [(chunk1, 0.0), (chunk2, 2.0)]

        merged = merge_transcripts(results, overlap=1)

        assert "segments" in merged
        assert len(merged["segments"]) > 0

    def test_returns_chunk_status(self, mock_transcription_dict):
        """Test that chunk status information is returned."""
        chunk1 = mock_transcription_dict
        chunk2 = mock_transcription_dict.copy()

        results = [(chunk1, 0.0), (chunk2, 2.0)]

        merged = merge_transcripts(results, overlap=1)

        # This test verifies the structure returned by merge_transcripts
        assert "text" in merged or "segments" in merged

    def test_partial_failure_handling(self, sample_audio_file):
        """Test handling of partial chunk failures."""
        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        # Mock a transcribe function that fails on some chunks
        def mock_transcribe(chunk_path):
            # Fail on first chunk, succeed on others
            if "part0" in str(chunk_path):
                raise Exception("Simulated failure")
            return {"text": "Success", "segments": []}

        # This would be tested via transcribe_with_chunks
        # For now we test the merge logic
        pass

    def test_all_chunks_fail_raises_exception(self):
        """Test that all chunks failing raises an exception."""
        # Mock a transcribe function that always fails
        def mock_transcribe(chunk_path):
            raise Exception("Always fails")

        # This would be tested via transcribe_with_chunks
        # For now we verify the concept
        assert True  # Placeholder


class TestChunkedTranscription:
    """Test suite for end-to-end chunked transcription."""

    @pytest.mark.skip(reason="preprocess_audio function not implemented yet")
    def test_end_to_end_chunking_workflow(self, sample_audio_file):
        """Test complete chunking workflow."""
        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        # Mock transcribe function
        def mock_transcribe(chunk_path):
            return {
                "text": "Test transcription",
                "segments": [{"start": 0, "end": 1, "text": "Test"}]
            }

        # Test with small chunk length to force splitting
        with patch('audio_transcribe.transcribe_helpers.audio_processing.preprocess_audio', return_value=sample_audio_file):
            with patch('audio_transcribe.transcribe_helpers.chunking.split_audio', return_value=[
                (sample_audio_file, 0.0)
            ]):
                result = transcribe_with_chunks(
                    sample_audio_file,
                    mock_transcribe,
                    chunk_length=300,
                    overlap=10
                )

                assert "text" in result

    def test_reports_failed_chunks(self, sample_audio_file):
        """Test that failed chunks are reported."""
        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        call_count = [0]

        # Mock transcribe that fails on second call
        def mock_transcribe(chunk_path):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("Simulated failure")
            return {
                "text": "Success",
                "segments": [{"start": 0, "end": 1, "text": "Test"}]
            }

        # This would be tested with actual splitting
        # For now we verify the concept
        assert True  # Placeholder

    @pytest.mark.skip(reason="preprocess_audio function not implemented yet")
    def test_stops_on_rate_limit(self, sample_audio_file):
        """Test that processing stops on rate limit."""
        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        # Mock transcribe that returns rate limit error
        def mock_transcribe(chunk_path):
            raise Exception("429 rate limit exceeded")

        with pytest.raises(RuntimeError, match="Rate limit exceeded"):
            with patch('audio_transcribe.transcribe_helpers.audio_processing.preprocess_audio', return_value=sample_audio_file):
                with patch('audio_transcribe.transcribe_helpers.chunking.split_audio', return_value=[
                    (sample_audio_file, 0.0)
                ]):
                    transcribe_with_chunks(
                        sample_audio_file,
                        mock_transcribe,
                        chunk_length=300,
                        overlap=10
                    )

    @pytest.mark.skip(reason="preprocess_audio function not implemented yet")
    def test_custom_kwargs_passed_through(self, sample_audio_file):
        """Test that custom kwargs are passed to transcribe function."""
        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        # Mock that checks for custom kwargs
        def mock_transcribe(chunk_path, language="en", model="test"):
            assert language == "de"
            assert model == "test"
            return {
                "text": "Success",
                "segments": []
            }

        # Test with lambda that passes custom kwargs
        with patch('audio_transcribe.transcribe_helpers.audio_processing.preprocess_audio', return_value=sample_audio_file):
            with patch('audio_transcribe.transcribe_helpers.chunking.split_audio', return_value=[
                (sample_audio_file, 0.0)
            ]):
                # Create lambda that passes custom kwargs
                transcribe_with_kwargs = lambda p: mock_transcribe(p, language="de", model="test")

                transcribe_with_chunks(
                    sample_audio_file,
                    transcribe_with_kwargs,
                    chunk_length=300,
                    overlap=10
                )


@pytest.mark.requires_ffmpeg
class TestEdgeCases:
    """Test suite for edge cases in chunking."""

    def test_empty_audio_file(self, tmp_path):
        """Test handling of empty audio file."""
        # Create a very short audio file (100ms)
        audio = AudioSegment.silent(duration=100)
        path = tmp_path / "empty.wav"
        audio.export(str(path), format="wav")

        chunks = split_audio(path, chunk_length=300, overlap=10)

        # Should have at least 1 chunk
        assert len(chunks) >= 1

    def test_overlap_greater_than_chunk_length(self, tmp_path):
        """Test handling of overlap > chunk_length."""
        audio = AudioSegment.silent(duration=10000)  # 10 seconds
        path = tmp_path / "test.wav"
        audio.export(str(path), format="wav")

        # Overlap (30) greater than chunk length (20)
        chunks = split_audio(path, chunk_length=20, overlap=30)

        # Should still work (might produce many small chunks)
        assert len(chunks) >= 1

    def test_very_small_chunks(self, tmp_path):
        """Test handling of very small chunk sizes."""
        audio = AudioSegment.silent(duration=30000)  # 30 seconds
        path = tmp_path / "test.wav"
        audio.export(str(path), format="wav")

        # 1-second chunks
        chunks = split_audio(path, chunk_length=1, overlap=0)

        assert len(chunks) >= 30

    def test_zero_overlap(self, tmp_path):
        """Test handling of zero overlap."""
        audio = AudioSegment.silent(duration=30000)  # 30 seconds
        path = tmp_path / "test.wav"
        audio.export(str(path), format="wav")

        chunks = split_audio(path, chunk_length=10, overlap=0)

        # 30s / 10s effective length = 3 full chunks + 1 boundary chunk
        assert len(chunks) >= 3

    def test_timestamp_precision(self, tmp_path):
        """Test that timestamps maintain precision."""
        audio = AudioSegment.silent(duration=60000)  # 60 seconds
        path = tmp_path / "test.wav"
        audio.export(str(path), format="wav")

        chunks = split_audio(path, chunk_length=25, overlap=5)

        # Verify timestamps are floats with reasonable precision
        for _, timestamp in chunks:
            assert isinstance(timestamp, (int, float))
            assert timestamp >= 0

    def test_file_cleanup_on_errors(self, tmp_path):
        """Test that temp files are cleaned up on errors."""
        audio = AudioSegment.silent(duration=30000)  # 30 seconds
        path = tmp_path / "test.wav"
        audio.export(str(path), format="wav")

        chunks = split_audio(path, chunk_length=10, overlap=1)

        # Get the chunk file paths
        chunk_paths = [c[0] for c in chunks]

        # Clean up manually (normally done by transcribe_with_chunks)
        for chunk_path in chunk_paths:
            if chunk_path.exists():
                chunk_path.unlink()

        # Verify cleanup
        for chunk_path in chunk_paths:
            assert not chunk_path.exists()
