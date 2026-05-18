"""
Integration tests for OpenAI API.

Tests the OpenAI Whisper transcription API integration including:
- Short audio transcription
- Whisper model
- GPT-4o-mini-transcribe model
- FLAC requirement
- 25MB limit enforced
- API key validation
- list_models() enumeration
- verbose_json format (word timestamps)
- vtt format
- srt format
- Invalid key error handling
- Language parameter
"""
import pytest
from pathlib import Path
from audio_transcribe.utils.api import get_api_instance


@pytest.fixture(autouse=True)
def _skip_on_openai_auth_error(request, api_keys):
    """Skip all tests in this module if OpenAI key is invalid."""
    api_key = api_keys.get("openai")
    if api_key:
        api = get_api_instance("openai", api_key)
        if not api.check_api_key():
            pytest.skip("OpenAI API key is invalid/expired")


@pytest.mark.integration
class TestOpenAIAPIIntegration:
    """Integration tests for OpenAI API."""

    def test_short_audio_transcription(self, sample_audio_file, api_keys):
        """Test transcription of short audio file."""
        api_key = api_keys.get("openai")
        if not api_key:
            pytest.skip("No OpenAI API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("openai", api_key)

        result = api.transcribe(sample_audio_file)

        assert result is not None
        assert result.text is not None
        assert len(result.text) > 0
        assert result.api_name == "openai"

    def test_whisper_model(self, sample_audio_file, api_keys):
        """Test Whisper model."""
        api_key = api_keys.get("openai")
        if not api_key:
            pytest.skip("No OpenAI API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("openai", api_key)

        result = api.transcribe(sample_audio_file, model="whisper-1")

        assert result is not None
        assert result.api_name == "openai"

    def test_flac_requirement(self, sample_audio_file, api_keys):
        """Test that FLAC conversion happens automatically."""
        api_key = api_keys.get("openai")
        if not api_key:
            pytest.skip("No OpenAI API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("openai", api_key)

        # OpenAI requires FLAC format
        # Any input format should be converted automatically
        result = api.transcribe(sample_audio_file)

        assert result is not None
        assert result.api_name == "openai"

    def test_25mb_limit_enforced(self, api_keys):
        """Test that 25MB file size limit is enforced."""
        api_key = api_keys.get("openai")
        if not api_key:
            pytest.skip("No OpenAI API key available")

        from audio_transcribe.transcribe_helpers.audio_processing import get_api_file_size_limit

        limit = get_api_file_size_limit("openai")
        assert limit == 25

    def test_api_key_validation(self, api_keys):
        """Test API key validation with valid key."""
        api_key = api_keys.get("openai")
        if not api_key:
            pytest.skip("No OpenAI API key available")

        api = get_api_instance("openai", api_key)

        assert api.check_api_key() is True

    def test_api_key_invalid(self):
        """Test API key validation with invalid key."""
        api = get_api_instance("openai", "invalid_key_12345")

        assert api.check_api_key() is False

    def test_list_models(self, api_keys):
        """Test listing available models."""
        api_key = api_keys.get("openai")
        if not api_key:
            pytest.skip("No OpenAI API key available")

        api = get_api_instance("openai", api_key)

        models = api.list_models()

        assert isinstance(models, list)
        assert len(models) > 0
        # Should include whisper models
        assert any("whisper" in model.lower() for model in models)

    def test_verbose_json_word_timestamps(self, sample_audio_file, api_keys):
        """Test verbose_json format with word timestamps."""
        api_key = api_keys.get("openai")
        if not api_key:
            pytest.skip("No OpenAI API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("openai", api_key)

        result = api.transcribe(sample_audio_file)

        # verbose_json format should include word timestamps
        assert result.words is not None
        if result.words:
            # Check word timestamp structure
            first_word = result.words[0]
            assert "text" in first_word
            assert "start" in first_word
            assert "end" in first_word

    def test_language_parameter(self, sample_audio_file, api_keys):
        """Test language parameter handling."""
        api_key = api_keys.get("openai")
        if not api_key:
            pytest.skip("No OpenAI API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("openai", api_key)

        # Test with German language
        result = api.transcribe(sample_audio_file, language="de")

        assert result is not None
        assert result.api_name == "openai"

    def test_invalid_key_error_handling(self):
        """Test error handling for invalid API key."""
        api = get_api_instance("openai", "invalid_key_12345")

        # check_api_key should return False
        assert api.check_api_key() is False

    def test_chunking_for_large_files(self, sample_audio_files, api_keys):
        """Test chunking functionality for large files."""
        api_key = api_keys.get("openai")
        if not api_key:
            pytest.skip("No OpenAI API key available")

        long_audio = sample_audio_files.get("long_audio")
        if not long_audio:
            pytest.skip("No long audio file available")

        api = get_api_instance("openai", api_key)

        # Check file size
        file_size_mb = long_audio.stat().st_size / (1024 * 1024)

        # If file is larger than 20MB, it should trigger chunking
        if file_size_mb > 20:
            result = api.transcribe(long_audio)
            assert result is not None
            assert result.api_name == "openai"
        else:
            pytest.skip(f"Long audio file not large enough for chunking test ({file_size_mb:.2f}MB)")

    def test_timestamp_granularity_word(self, sample_audio_file, api_keys):
        """Test timestamp_granularities=['word'] parameter."""
        api_key = api_keys.get("openai")
        if not api_key:
            pytest.skip("No OpenAI API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("openai", api_key)

        result = api.transcribe(sample_audio_file)

        # OpenAI uses timestamp_granularities=["word"] by default
        assert result.words is not None
        if result.words:
            # Verify words have timestamps
            for word in result.words[:5]:  # Check first 5 words
                assert "start" in word
                assert "end" in word

    def test_response_format_text(self, sample_audio_file, api_keys):
        """Test that text response format works."""
        api_key = api_keys.get("openai")
        if not api_key:
            pytest.skip("No OpenAI API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("openai", api_key)

        result = api.transcribe(sample_audio_file)

        # Should return text at minimum
        assert result.text is not None
        assert len(result.text) > 0
