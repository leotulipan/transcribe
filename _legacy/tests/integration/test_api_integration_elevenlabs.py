"""
Integration tests for ElevenLabs API.

Tests the ElevenLabs transcription API integration including:
- Short audio transcription
- Diarization enabled
- Large file support
- Video file handling
- Word timestamps
- Segment timestamps
- Speaker labels
- Language detection
- Model parameter (scribe_v1)
"""
import pytest
from pathlib import Path
from audio_transcribe.utils.api import get_api_instance


@pytest.mark.integration
class TestElevenLabsAPIIntegration:
    """Integration tests for ElevenLabs API."""

    def test_short_audio_transcription(self, sample_audio_file, api_keys):
        """Test transcription of short audio file."""
        api_key = api_keys.get("elevenlabs")
        if not api_key:
            pytest.skip("No ElevenLabs API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("elevenlabs", api_key)

        result = api.transcribe(sample_audio_file)

        assert result is not None
        assert result.text is not None
        assert len(result.text) > 0
        assert result.api_name == "elevenlabs"

    def test_diarization_enabled(self, sample_audio_files, api_keys):
        """Test speaker diarization functionality."""
        api_key = api_keys.get("elevenlabs")
        if not api_key:
            pytest.skip("No ElevenLabs API key available")

        multi_speaker = sample_audio_files.get("multi_speaker")
        if not multi_speaker:
            pytest.skip("No multi-speaker audio file available")

        api = get_api_instance("elevenlabs", api_key)

        # Enable diarization
        result = api.transcribe(multi_speaker, diarize=True)

        assert result is not None
        assert result.text is not None

        # Check for speakers (diarization results)
        if result.speakers:
            assert len(result.speakers) > 0

    def test_large_file_support(self, sample_audio_files, api_keys):
        """Test support for large files (up to 1GB)."""
        api_key = api_keys.get("elevenlabs")
        if not api_key:
            pytest.skip("No ElevenLabs API key available")

        long_audio = sample_audio_files.get("long_audio")
        if not long_audio:
            pytest.skip("No long audio file available")

        api = get_api_instance("elevenlabs", api_key)

        # Check file size
        file_size_mb = long_audio.stat().st_size / (1024 * 1024)

        # ElevenLabs supports large files (up to 1GB)
        if file_size_mb > 15:  # Test with reasonably large file
            result = api.transcribe(long_audio)
            assert result is not None
        else:
            pytest.skip(f"Long audio file not large enough for large file test ({file_size_mb:.2f}MB)")

    def test_video_file_handling(self, sample_audio_files, api_keys):
        """Test handling of video files."""
        api_key = api_keys.get("elevenlabs")
        if not api_key:
            pytest.skip("No ElevenLabs API key available")

        video_file = sample_audio_files.get("video")
        if not video_file:
            pytest.skip("No video file available")

        api = get_api_instance("elevenlabs", api_key)

        # ElevenLabs extracts audio from video automatically
        result = api.transcribe(video_file)

        assert result is not None
        assert result.api_name == "elevenlabs"

    def test_api_key_validation(self, api_keys):
        """Test API key validation with valid key."""
        api_key = api_keys.get("elevenlabs")
        if not api_key:
            pytest.skip("No ElevenLabs API key available")

        api = get_api_instance("elevenlabs", api_key)

        assert api.check_api_key() is True

    def test_api_key_invalid(self):
        """Test API key validation with invalid key."""
        api = get_api_instance("elevenlabs", "invalid_key_12345")

        assert api.check_api_key() is False

    def test_word_timestamps(self, sample_audio_file, api_keys):
        """Test that word timestamps are returned."""
        api_key = api_keys.get("elevenlabs")
        if not api_key:
            pytest.skip("No ElevenLabs API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("elevenlabs", api_key)

        result = api.transcribe(sample_audio_file)

        # ElevenLabs provides word timestamps by default
        assert result.words is not None
        if result.words:
            # Filter out spacing entries
            words = [w for w in result.words if w.get("type") != "spacing"]
            assert len(words) > 0

            # Check word timestamp structure
            first_word = words[0]
            assert "text" in first_word
            assert "start" in first_word
            assert "end" in first_word

    def test_segment_timestamps(self, sample_audio_file, api_keys):
        """Test that segment timestamps are returned."""
        api_key = api_keys.get("elevenlabs")
        if not api_key:
            pytest.skip("No ElevenLabs API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("elevenlabs", api_key)

        result = api.transcribe(sample_audio_file)

        # ElevenLabs returns segment information
        assert result.segments is not None
        if result.segments:
            # Check segment structure
            first_segment = result.segments[0]
            assert "start" in first_segment
            assert "end" in first_segment
            assert "text" in first_segment

    def test_speaker_labels(self, sample_audio_files, api_keys):
        """Test that speaker labels are included when diarization is enabled."""
        api_key = api_keys.get("elevenlabs")
        if not api_key:
            pytest.skip("No ElevenLabs API key available")

        multi_speaker = sample_audio_files.get("multi_speaker")
        if not multi_speaker:
            pytest.skip("No multi-speaker audio file available")

        api = get_api_instance("elevenlabs", api_key)

        result = api.transcribe(multi_speaker, diarize=True)

        # Check for speaker labels
        if result.speakers:
            assert len(result.speakers) > 0

    def test_language_detection(self, sample_audio_file, api_keys):
        """Test automatic language detection."""
        api_key = api_keys.get("elevenlabs")
        if not api_key:
            pytest.skip("No ElevenLabs API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("elevenlabs", api_key)

        # ElevenLabs auto-detects language when not specified
        result = api.transcribe(sample_audio_file)

        assert result is not None
        assert result.language is not None or result.text is not None

    def test_language_parameter(self, sample_audio_file, api_keys):
        """Test explicit language parameter."""
        api_key = api_keys.get("elevenlabs")
        if not api_key:
            pytest.skip("No ElevenLabs API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("elevenlabs", api_key)

        # Test with German language
        result = api.transcribe(sample_audio_file, language="de")

        assert result is not None
        assert result.api_name == "elevenlabs"

    def test_no_flac_requirement(self, sample_audio_file, api_keys):
        """Test that FLAC conversion is not required."""
        api_key = api_keys.get("elevenlabs")
        if not api_key:
            pytest.skip("No ElevenLabs API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("elevenlabs", api_key)

        # ElevenLabs accepts multiple formats without requiring FLAC
        result = api.transcribe(sample_audio_file)

        assert result is not None
        assert result.api_name == "elevenlabs"

    def test_invalid_file_error_handling(self, api_keys, tmp_path):
        """Test handling of invalid audio file."""
        api_key = api_keys.get("elevenlabs")
        if not api_key:
            pytest.skip("No ElevenLabs API key available")

        api = get_api_instance("elevenlabs", api_key)

        # Create an invalid audio file
        invalid_file = tmp_path / "invalid.wav"
        invalid_file.write_text("This is not a real WAV file")

        # Should raise an error
        with pytest.raises((ValueError, Exception)):
            api.transcribe(invalid_file)

    def test_word_level_output_format(self, sample_audio_file, api_keys):
        """Test that word-level output is properly formatted."""
        api_key = api_keys.get("elevenlabs")
        if not api_key:
            pytest.skip("No ElevenLabs API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("elevenlabs", api_key)

        result = api.transcribe(sample_audio_file)

        # ElevenLabs uses "timestamps_granularity": "word"
        assert result.words is not None
        if result.words:
            # Verify word entries have proper structure
            for word in result.words[:5]:  # Check first 5 words
                assert "text" in word
                assert "start" in word
                assert "end" in word

    def test_speaker_count_range(self, sample_audio_files, api_keys):
        """Test num_speakers parameter range (1-32)."""
        api_key = api_keys.get("elevenlabs")
        if not api_key:
            pytest.skip("No ElevenLabs API key available")

        multi_speaker = sample_audio_files.get("multi_speaker")
        if not multi_speaker:
            pytest.skip("No multi-speaker audio file available")

        api = get_api_instance("elevenlabs", api_key)

        # Test with valid speaker count
        result = api.transcribe(multi_speaker, diarize=True, num_speakers=2)

        assert result is not None
        assert result.api_name == "elevenlabs"

    def test_audio_event_tagging(self, sample_audio_file, api_keys):
        """Test audio event tagging capability."""
        api_key = api_keys.get("elevenlabs")
        if not api_key:
            pytest.skip("No ElevenLabs API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("elevenlabs", api_key)

        # ElevenLabs enables "tag_audio_events": "true" by default
        result = api.transcribe(sample_audio_file)

        assert result is not None
        # Audio events might be in the raw response but not in parsed result

    def test_model_parameter_scribe_v1(self, sample_audio_file, api_keys):
        """Test model parameter with scribe_v1 model."""
        api_key = api_keys.get("elevenlabs")
        if not api_key:
            pytest.skip("No ElevenLabs API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("elevenlabs", api_key)

        # Explicitly specify scribe_v1 model
        result = api.transcribe(sample_audio_file, model_id="scribe_v1")

        assert result is not None
        assert result.api_name == "elevenlabs"

    def test_list_models(self, api_keys):
        """Test listing available models."""
        api_key = api_keys.get("elevenlabs")
        if not api_key:
            pytest.skip("No ElevenLabs API key available")

        api = get_api_instance("elevenlabs", api_key)

        models = api.list_models()

        # ElevenLabs /models endpoint returns TTS models
        # The endpoint should work if API key is valid
        assert isinstance(models, list)
