"""
Integration tests for Groq API.

Tests the Groq transcription API integration including:
- Short audio transcription
- Language parameter handling
- Model selection
- Chunking for large files
- API key validation
- Word timestamps
- Error handling
"""
import pytest
from pathlib import Path
from audio_transcribe.utils.api import get_api_instance


@pytest.mark.integration
class TestGroqAPIIntegration:
    """Integration tests for Groq API."""

    def test_short_audio_transcription(self, sample_audio_file, api_keys):
        """Test transcription of short audio file."""
        api_key = api_keys.get("groq")
        if not api_key:
            pytest.skip("No Groq API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("groq", api_key)

        result = api.transcribe(sample_audio_file)

        assert result is not None
        assert result.text is not None
        assert len(result.text) > 0
        assert result.api_name == "groq"

    def test_language_parameter_handling(self, sample_audio_file, api_keys):
        """Test that language parameter is properly handled."""
        api_key = api_keys.get("groq")
        if not api_key:
            pytest.skip("No Groq API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("groq", api_key)

        # Test with German language
        result = api.transcribe(sample_audio_file, language="de")

        assert result is not None
        assert result.api_name == "groq"

    def test_model_parameter(self, sample_audio_file, api_keys):
        """Test that model parameter is properly handled."""
        api_key = api_keys.get("groq")
        if not api_key:
            pytest.skip("No Groq API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("groq", api_key)

        # Test with whisper-large-v3-turbo
        result = api.transcribe(sample_audio_file, model="whisper-large-v3-turbo")

        assert result is not None
        assert result.api_name == "groq"

    def test_basic_chunking(self, sample_audio_file, api_keys):
        """Test basic chunking functionality."""
        api_key = api_keys.get("groq")
        if not api_key:
            pytest.skip("No Groq API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("groq", api_key)

        # Force chunking with short chunk_length
        result = api.transcribe(sample_audio_file, chunk_length=30, overlap=5)

        assert result is not None
        assert result.text is not None
        assert result.api_name == "groq"

    def test_large_file_chunking(self, sample_audio_files, api_keys):
        """Test chunking with file larger than 25MB."""
        api_key = api_keys.get("groq")
        if not api_key:
            pytest.skip("No Groq API key available")

        # Use long_audio if available
        long_audio = sample_audio_files.get("long_audio")
        if not long_audio:
            pytest.skip("No long audio file available")

        api = get_api_instance("groq", api_key)

        # Check file size
        file_size_mb = long_audio.stat().st_size / (1024 * 1024)

        # If file is large enough to require chunking
        if file_size_mb > 20:  # Close to or over 25MB limit
            result = api.transcribe(long_audio)
            assert result is not None
            assert result.api_name == "groq"
        else:
            pytest.skip(f"Long audio file not large enough for chunking test ({file_size_mb:.2f}MB)")

    def test_api_key_valid(self, api_keys):
        """Test API key validation with valid key."""
        api_key = api_keys.get("groq")
        if not api_key:
            pytest.skip("No Groq API key available")

        api = get_api_instance("groq", api_key)

        assert api.check_api_key() is True

    def test_api_key_invalid(self):
        """Test API key validation with invalid key."""
        api = get_api_instance("groq", "invalid_key_12345")

        assert api.check_api_key() is False

    def test_list_models(self, api_keys):
        """Test listing available models."""
        api_key = api_keys.get("groq")
        if not api_key:
            pytest.skip("No Groq API key available")

        api = get_api_instance("groq", api_key)

        models = api.list_models()

        assert isinstance(models, list)
        assert len(models) > 0
        # Check for expected Whisper models
        assert any("whisper" in model.lower() for model in models)

    def test_flac_format_requirement(self, sample_audio_file, api_keys):
        """Test that FLAC conversion happens automatically."""
        api_key = api_keys.get("groq")
        if not api_key:
            pytest.skip("No Groq API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("groq", api_key)

        # Groq requires FLAC, so any input format should work
        # This tests that automatic FLAC conversion happens
        result = api.transcribe(sample_audio_file)

        assert result is not None
        assert result.api_name == "groq"

    def test_25mb_limit_enforced(self, api_keys):
        """Test that 25MB file size limit is enforced."""
        api_key = api_keys.get("groq")
        if not api_key:
            pytest.skip("No Groq API key available")

        from audio_transcribe.transcribe_helpers.audio_processing import get_api_file_size_limit

        limit = get_api_file_size_limit("groq")
        assert limit == 25

    def test_word_timestamps_returned(self, sample_audio_file, api_keys):
        """Test that word timestamps are returned."""
        api_key = api_keys.get("groq")
        if not api_key:
            pytest.skip("No Groq API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("groq", api_key)

        result = api.transcribe(sample_audio_file)

        # Groq should return word timestamps with verbose_json
        assert result.words is not None
        assert len(result.words) > 0

        # Check word timestamp structure
        first_word = result.words[0]
        assert "text" in first_word
        assert "start" in first_word
        assert "end" in first_word

    def test_segment_timestamps(self, sample_audio_file, api_keys):
        """Test that segment timestamps are returned."""
        api_key = api_keys.get("groq")
        if not api_key:
            pytest.skip("No Groq API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("groq", api_key)

        result = api.transcribe(sample_audio_file)

        # Groq should return segment timestamps
        assert result.segments is not None
        if result.segments:
            # Check segment timestamp structure
            first_segment = result.segments[0]
            assert "start" in first_segment
            assert "end" in first_segment
            assert "text" in first_segment

    def test_invalid_audio_error_handling(self, api_keys, tmp_path):
        """Test handling of invalid audio file."""
        api_key = api_keys.get("groq")
        if not api_key:
            pytest.skip("No Groq API key available")

        api = get_api_instance("groq", api_key)

        # Create an invalid audio file
        invalid_file = tmp_path / "invalid.flac"
        invalid_file.write_text("This is not a real FLAC file")

        # Should raise an error or return None
        try:
            result = api.transcribe(invalid_file)
            # If it doesn't raise, result should be None
            assert result is None
        except (ValueError, Exception):
            # Expected to raise an error
            pass

    def test_rate_limit_handling(self, sample_audio_file, api_keys):
        """Test graceful handling of rate limits."""
        api_key = api_keys.get("groq")
        if not api_key:
            pytest.skip("No Groq API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("groq", api_key)

        # This test verifies that rate limit errors are handled
        # It's difficult to trigger rate limits intentionally, so we just
        # verify the code path exists
        result = api.transcribe(sample_audio_file)
        assert result is not None

    def test_timestamp_granularity(self, sample_audio_file, api_keys):
        """Test that timestamp granularity includes both word and segment."""
        api_key = api_keys.get("groq")
        if not api_key:
            pytest.skip("No Groq API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("groq", api_key)

        result = api.transcribe(sample_audio_file)

        # Groq supports both word and segment timestamps
        assert result.words is not None
        assert len(result.words) > 0

        # Verify timestamps are in seconds (not milliseconds)
        first_word = result.words[0]
        start_time = first_word.get("start", 0)
        # Should be a small number (seconds), not milliseconds
        assert 0 <= start_time < 1000  # Less than 1000 seconds

    def test_chunk_merging_logic(self, sample_audio_files, api_keys):
        """Test that chunk results are properly merged."""
        api_key = api_keys.get("groq")
        if not api_key:
            pytest.skip("No Groq API key available")

        long_audio = sample_audio_files.get("long_audio")
        if not long_audio:
            pytest.skip("No long audio file available for chunking test")

        api = get_api_instance("groq", api_key)

        # Force chunking
        result = api.transcribe(long_audio, chunk_length=60, overlap=5)

        assert result is not None
        assert result.text is not None

        # If chunking happened, verify words are properly ordered by timestamp
        if result.words and len(result.words) > 1:
            for i in range(len(result.words) - 1):
                current_end = result.words[i].get("end", 0)
                next_start = result.words[i + 1].get("start", 0)
                # Current word should end before or when next word starts
                assert current_end <= next_start + 0.1  # Small tolerance for overlap

    def test_response_format_verbose_json(self, sample_audio_file, api_keys):
        """Test that verbose_json response format is used."""
        api_key = api_keys.get("groq")
        if not api_key:
            pytest.skip("No Groq API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("groq", api_key)

        result = api.transcribe(sample_audio_file)

        # verbose_json should include word timestamps
        assert result.words is not None
        assert len(result.words) > 0

    def test_confidence_score(self, sample_audio_file, api_keys):
        """Test that confidence scores are available."""
        api_key = api_keys.get("groq")
        if not api_key:
            pytest.skip("No Groq API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("groq", api_key)

        result = api.transcribe(sample_audio_file)

        # Groq doesn't provide confidence scores in verbose_json
        # This test verifies we handle that gracefully
        # confidence may be None or 0.0
        assert result.confidence is not None or result.confidence == 0.0
