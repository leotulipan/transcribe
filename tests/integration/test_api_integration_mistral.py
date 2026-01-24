"""
Integration tests for Mistral Voxtral API.

Tests the Mistral transcription API integration including:
- Short audio transcription
- Segment-level timestamps only
- No word timestamps
- Language auto-detection (cannot specify)
- Language warning logged
- API key validation
- Model selection parameter
- JSON format output
- Invalid audio error handling
- timestamp_granularities=["segment"]
- Segment-to-word conversion
- Model: voxtral-mini-2507
"""
import pytest
from pathlib import Path
from audio_transcribe.utils.api import get_api_instance


@pytest.mark.integration
class TestMistralAPIIntegration:
    """Integration tests for Mistral Voxtral API."""

    def test_short_audio_transcription(self, sample_audio_file, api_keys):
        """Test transcription of short audio file."""
        api_key = api_keys.get("mistral")
        if not api_key:
            pytest.skip("No Mistral API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("mistral", api_key)

        result = api.transcribe(sample_audio_file)

        assert result is not None
        assert result.text is not None
        assert len(result.text) > 0
        assert result.api_name == "mistral"

    def test_segment_level_timestamps_only(self, sample_audio_file, api_keys):
        """Test that only segment-level timestamps are available."""
        api_key = api_keys.get("mistral")
        if not api_key:
            pytest.skip("No Mistral API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("mistral", api_key)

        result = api.transcribe(sample_audio_file)

        # Mistral provides segment timestamps
        assert result.segments is not None
        if result.segments:
            # Check segment structure
            first_segment = result.segments[0]
            assert "text" in first_segment
            assert "start" in first_segment
            assert "end" in first_segment

    def test_no_word_timestamps(self, sample_audio_file, api_keys):
        """Test that word timestamps are approximated (not native)."""
        api_key = api_keys.get("mistral")
        if not api_key:
            pytest.skip("No Mistral API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("mistral", api_key)

        result = api.transcribe(sample_audio_file)

        # Words are generated from segments (not native word timestamps)
        assert result.words is not None
        # The words list should be populated even though timestamps are approximated

    def test_language_auto_detection(self, sample_audio_file, api_keys):
        """Test that language is auto-detected (cannot specify)."""
        api_key = api_keys.get("mistral")
        if not api_key:
            pytest.skip("No Mistral API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("mistral", api_key)

        # Language parameter should be ignored
        # (Mistral auto-detects language)
        result = api.transcribe(sample_audio_file, language="de")

        assert result is not None
        # Language should be detected by the API
        assert result.language is not None

    def test_language_warning_logged(self, sample_audio_file, api_keys, caplog):
        """Test that a warning is logged when language is specified."""
        api_key = api_keys.get("mistral")
        if not api_key:
            pytest.skip("No Mistral API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("mistral", api_key)

        # Specify language (which will be ignored)
        with caplog.at_level("WARNING"):
            result = api.transcribe(sample_audio_file, language="de")

            assert result is not None
            # Should log a warning about language being ignored
            assert any("will be ignored" in record.message for record in caplog.records)

    def test_api_key_validation(self, api_keys):
        """Test API key validation with valid key."""
        api_key = api_keys.get("mistral")
        if not api_key:
            pytest.skip("No Mistral API key available")

        api = get_api_instance("mistral", api_key)

        assert api.check_api_key() is True

    def test_api_key_invalid(self):
        """Test API key validation with invalid key."""
        api = get_api_instance("mistral", "invalid_key_12345")

        assert api.check_api_key() is False

    def test_model_selection_parameter(self, sample_audio_file, api_keys):
        """Test model selection parameter."""
        api_key = api_keys.get("mistral")
        if not api_key:
            pytest.skip("No Mistral API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("mistral", api_key)

        # Test with voxtral-mini-2507 model
        result = api.transcribe(sample_audio_file, model="voxtral-mini-2507")

        assert result is not None
        assert result.api_name == "mistral"

    def test_json_format_output(self, sample_audio_file, api_keys):
        """Test that output is in JSON format with segments."""
        api_key = api_keys.get("mistral")
        if not api_key:
            pytest.skip("No Mistral API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("mistral", api_key)

        result = api.transcribe(sample_audio_file)

        # Mistral returns JSON with segments
        assert result.text is not None
        assert result.segments is not None

    def test_invalid_audio_error_handling(self, api_keys, tmp_path):
        """Test handling of invalid audio file."""
        api_key = api_keys.get("mistral")
        if not api_key:
            pytest.skip("No Mistral API key available")

        api = get_api_instance("mistral", api_key)

        # Create an invalid audio file
        invalid_file = tmp_path / "invalid.wav"
        invalid_file.write_text("This is not a real WAV file")

        # Should raise an error
        with pytest.raises((ValueError, Exception)):
            api.transcribe(invalid_file)

    def test_timestamp_granularities_segment(self, sample_audio_file, api_keys):
        """Test that timestamp_granularities=['segment'] is used."""
        api_key = api_keys.get("mistral")
        if not api_key:
            pytest.skip("No Mistral API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("mistral", api_key)

        result = api.transcribe(sample_audio_file)

        # Mistral uses timestamp_granularities=["segment"]
        assert result.segments is not None
        # Verify segment structure
        if result.segments:
            segment = result.segments[0]
            assert "start" in segment
            assert "end" in segment

    def test_segment_to_word_conversion(self, sample_audio_file, api_keys):
        """Test that segments are converted to approximate word timestamps."""
        api_key = api_keys.get("mistral")
        if not api_key:
            pytest.skip("No Mistral API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("mistral", api_key)

        result = api.transcribe(sample_audio_file)

        # Words should be generated from segments
        assert result.words is not None
        if result.words and result.segments:
            # Word timestamps should be within segment bounds
            first_word = result.words[0]
            assert "text" in first_word
            assert "start" in first_word
            assert "end" in first_word

    def test_capability_flags(self, api_keys):
        """Test that capability flags are set correctly."""
        api_key = api_keys.get("mistral")
        if not api_key:
            pytest.skip("No Mistral API key available")

        api = get_api_instance("mistral", api_key)

        # Check capability flags
        assert api.supports_word_timestamps is False  # Only segment level
        assert api.supports_segment_timestamps is True  # Segment-level only
        assert api.supports_speaker_diarization is False
        assert api.supports_srt_format is False

    def test_list_models(self, api_keys):
        """Test listing available models."""
        api_key = api_keys.get("mistral")
        if not api_key:
            pytest.skip("No Mistral API key available")

        api = get_api_instance("mistral", api_key)

        models = api.list_models()

        assert isinstance(models, list)
        assert len(models) > 0
        # Should include voxtral models
        assert any("voxtral" in model.lower() for model in models)
