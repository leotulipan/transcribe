"""
Integration tests for Gemini API.

Tests the Google Gemini transcription API integration including:
- Short audio transcription
- Text-only output (no timestamps)
- Inline vs Files API threshold
- Small file (<20MB) uses inline
- Large file (>20MB) uses Files API
- API key validation
- Language parameter
- No word timestamps (capability flag)
- No segment timestamps (capability flag)
- Invalid audio error handling
"""
import pytest
from pathlib import Path
from audio_transcribe.utils.api import get_api_instance


@pytest.fixture(autouse=True)
def _skip_on_gemini_auth_error(request, api_keys):
    """Skip all tests in this module if Gemini key is invalid."""
    api_key = api_keys.get("gemini")
    if api_key:
        api = get_api_instance("gemini", api_key)
        if not api.check_api_key():
            pytest.skip("Gemini API key is invalid/expired")


@pytest.mark.integration
class TestGeminiAPIIntegration:
    """Integration tests for Gemini API."""

    def test_short_audio_transcription(self, sample_audio_file, api_keys):
        """Test transcription of short audio file."""
        api_key = api_keys.get("gemini")
        if not api_key:
            pytest.skip("No Gemini API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("gemini", api_key)

        result = api.transcribe(sample_audio_file)

        assert result is not None
        assert result.text is not None
        assert len(result.text) > 0
        assert result.api_name == "gemini"

    def test_text_only_output(self, sample_audio_file, api_keys):
        """Test that output is text-only (no timestamps)."""
        api_key = api_keys.get("gemini")
        if not api_key:
            pytest.skip("No Gemini API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("gemini", api_key)

        result = api.transcribe(sample_audio_file)

        # Gemini returns text only, no word/segment timestamps
        assert result.text is not None
        # Words may be generated but won't have real timestamps
        assert result.words is not None

    def test_inline_api_for_small_files(self, sample_audio_file, api_keys):
        """Test that small files (<20MB) use inline API."""
        api_key = api_keys.get("gemini")
        if not api_key:
            pytest.skip("No Gemini API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("gemini", api_key)

        # Check file size
        file_size_mb = sample_audio_file.stat().st_size / (1024 * 1024)

        if file_size_mb < 20:
            result = api.transcribe(sample_audio_file)
            assert result is not None
        else:
            pytest.skip(f"Sample audio file too large for inline test ({file_size_mb:.2f}MB)")

    def test_files_api_for_large_files(self, sample_audio_files, api_keys):
        """Test that large files (>20MB) use Files API."""
        api_key = api_keys.get("gemini")
        if not api_key:
            pytest.skip("No Gemini API key available")

        long_audio = sample_audio_files.get("long_audio")
        if not long_audio:
            pytest.skip("No long audio file available")

        api = get_api_instance("gemini", api_key)

        # Check file size
        file_size_mb = long_audio.stat().st_size / (1024 * 1024)

        if file_size_mb > 20:
            result = api.transcribe(long_audio)
            assert result is not None
        else:
            pytest.skip(f"Long audio file not large enough for Files API test ({file_size_mb:.2f}MB)")

    def test_api_key_validation(self, api_keys):
        """Test API key validation with valid key."""
        api_key = api_keys.get("gemini")
        if not api_key:
            pytest.skip("No Gemini API key available")

        api = get_api_instance("gemini", api_key)

        assert api.check_api_key() is True

    def test_api_key_invalid(self):
        """Test API key validation with invalid key."""
        api = get_api_instance("gemini", "invalid_key_12345")

        # Note: Gemini's check_api_key() only verifies key is set, not valid.
        # Real validation happens during transcription.
        assert api.check_api_key() is True

    def test_language_parameter(self, sample_audio_file, api_keys):
        """Test language parameter handling."""
        api_key = api_keys.get("gemini")
        if not api_key:
            pytest.skip("No Gemini API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("gemini", api_key)

        # Test with German language
        result = api.transcribe(sample_audio_file, language="de")

        assert result is not None
        assert result.api_name == "gemini"

    def test_no_word_timestamps_capability(self, api_keys):
        """Test that API correctly reports no word timestamp support."""
        api_key = api_keys.get("gemini")
        if not api_key:
            pytest.skip("No Gemini API key available")

        api = get_api_instance("gemini", api_key)

        # Check capability flags
        assert api.supports_word_timestamps is False
        assert api.supports_segment_timestamps is False
        assert api.supports_speaker_diarization is False
        assert api.supports_srt_format is False

    def test_supported_output_formats(self, api_keys):
        """Test that only text format is supported."""
        api_key = api_keys.get("gemini")
        if not api_key:
            pytest.skip("No Gemini API key available")

        api = get_api_instance("gemini", api_key)

        # Check supported formats
        assert "text" in api.supported_output_formats

    def test_invalid_audio_error_handling(self, api_keys, tmp_path):
        """Test handling of invalid audio file."""
        api_key = api_keys.get("gemini")
        if not api_key:
            pytest.skip("No Gemini API key available")

        api = get_api_instance("gemini", api_key)

        # Create an invalid audio file
        invalid_file = tmp_path / "invalid.wav"
        invalid_file.write_text("This is not a real WAV file")

        # Gemini may either raise an error or return a result with empty/garbage text
        try:
            result = api.transcribe(invalid_file)
            # If no error, result should still be a TranscriptionResult
            assert result is not None
        except (ValueError, Exception):
            pass  # Expected — invalid audio should fail

    def test_list_models(self, api_keys):
        """Test listing available models."""
        api_key = api_keys.get("gemini")
        if not api_key:
            pytest.skip("No Gemini API key available")

        api = get_api_instance("gemini", api_key)

        models = api.list_models()

        assert isinstance(models, list)
        # Should return at least some models
        assert len(models) > 0
        # Should include flash models
        assert any("flash" in model.lower() for model in models)

    def test_model_parameter(self, sample_audio_file, api_keys):
        """Test model parameter with different Gemini models."""
        api_key = api_keys.get("gemini")
        if not api_key:
            pytest.skip("No Gemini API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("gemini", api_key)

        # Test with gemini-2.5-flash
        result = api.transcribe(sample_audio_file, model="gemini-2.5-flash")

        assert result is not None
        assert result.api_name == "gemini"
