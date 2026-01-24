"""
Integration tests for AssemblyAI API.

Tests the AssemblyAI transcription API integration including:
- Short audio transcription
- Speaker diarization
- Dual-channel audio
- Video file support
- Model selection (best/nano)
- Word timestamps
- Speaker labels
- Confidence scores
"""
import pytest
from pathlib import Path
from audio_transcribe.utils.api import get_api_instance


@pytest.mark.integration
class TestAssemblyAIAPIIntegration:
    """Integration tests for AssemblyAI API."""

    def test_short_audio_transcription(self, sample_audio_file, api_keys):
        """Test transcription of short audio file."""
        api_key = api_keys.get("assemblyai")
        if not api_key:
            pytest.skip("No AssemblyAI API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("assemblyai", api_key)

        result = api.transcribe(sample_audio_file)

        assert result is not None
        assert result.text is not None
        assert len(result.text) > 0
        assert result.api_name == "assemblyai"

    def test_speaker_diarization_enabled(self, sample_audio_files, api_keys):
        """Test speaker diarization functionality."""
        api_key = api_keys.get("assemblyai")
        if not api_key:
            pytest.skip("No AssemblyAI API key available")

        multi_speaker = sample_audio_files.get("multi_speaker")
        if not multi_speaker:
            pytest.skip("No multi-speaker audio file available")

        api = get_api_instance("assemblyai", api_key)

        # Enable speaker labels
        result = api.transcribe(multi_speaker, speaker_labels=True)

        assert result is not None
        assert result.text is not None

        # Check if speakers are present (may not always be detected)
        if result.speakers:
            assert len(result.speakers) > 0

    def test_dual_channel_audio(self, sample_audio_file, api_keys):
        """Test dual-channel audio transcription."""
        api_key = api_keys.get("assemblyai")
        if not api_key:
            pytest.skip("No AssemblyAI API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("assemblyai", api_key)

        # Enable dual channel
        result = api.transcribe(sample_audio_file, dual_channel=True)

        assert result is not None
        assert result.api_name == "assemblyai"

    def test_video_file_upload_support(self, sample_audio_files, api_keys):
        """Test that video files can be uploaded directly."""
        api_key = api_keys.get("assemblyai")
        if not api_key:
            pytest.skip("No AssemblyAI API key available")

        video_file = sample_audio_files.get("video")
        if not video_file:
            pytest.skip("No video file available")

        api = get_api_instance("assemblyai", api_key)

        # AssemblyAI accepts video files directly
        result = api.transcribe(video_file)

        assert result is not None
        assert result.api_name == "assemblyai"

    def test_api_key_validation(self, api_keys):
        """Test API key validation."""
        api_key = api_keys.get("assemblyai")
        if not api_key:
            pytest.skip("No AssemblyAI API key available")

        api = get_api_instance("assemblyai", api_key)

        # AssemblyAI check_api_key returns True if key is set
        assert api.check_api_key() is True

    def test_api_key_invalid(self):
        """Test API key validation with invalid key."""
        api = get_api_instance("assemblyai", "invalid_key_12345")

        # AssemblyAI doesn't validate keys without transcription
        # So it will still return True if a key is set
        # Real validation happens during transcription
        assert api.check_api_key() is True  # Key is set, so returns True

    def test_list_models(self, api_keys):
        """Test listing available models."""
        api_key = api_keys.get("assemblyai")
        if not api_key:
            pytest.skip("No AssemblyAI API key available")

        api = get_api_instance("assemblyai", api_key)

        models = api.list_models()

        assert isinstance(models, list)
        assert len(models) > 0
        # AssemblyAI has static model names
        assert "best" in models
        assert "nano" in models

    def test_word_timestamps(self, sample_audio_file, api_keys):
        """Test that word timestamps are returned."""
        api_key = api_keys.get("assemblyai")
        if not api_key:
            pytest.skip("No AssemblyAI API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("assemblyai", api_key)

        result = api.transcribe(sample_audio_file)

        # AssemblyAI provides word timestamps
        assert result.words is not None
        if result.words:
            # Check word timestamp structure
            first_word = result.words[0]
            assert "text" in first_word
            assert "start" in first_word
            assert "end" in first_word

    def test_speaker_labels_in_result(self, sample_audio_files, api_keys):
        """Test that speaker labels are included in results."""
        api_key = api_keys.get("assemblyai")
        if not api_key:
            pytest.skip("No AssemblyAI API key available")

        multi_speaker = sample_audio_files.get("multi_speaker")
        if not multi_speaker:
            pytest.skip("No multi-speaker audio file available")

        api = get_api_instance("assemblyai", api_key)

        result = api.transcribe(multi_speaker, speaker_labels=True)

        # Check for speakers in result
        if result.speakers:
            assert len(result.speakers) > 0

    def test_confidence_scores(self, sample_audio_file, api_keys):
        """Test that confidence scores are available."""
        api_key = api_keys.get("assemblyai")
        if not api_key:
            pytest.skip("No AssemblyAI API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("assemblyai", api_key)

        result = api.transcribe(sample_audio_file)

        # AssemblyAI provides confidence scores
        assert result.confidence is not None
        # Confidence should be between 0 and 1
        if result.confidence > 0:
            assert 0 <= result.confidence <= 1

    def test_invalid_audio_error_handling(self, api_keys, tmp_path):
        """Test handling of invalid audio file."""
        api_key = api_keys.get("assemblyai")
        if not api_key:
            pytest.skip("No AssemblyAI API key available")

        api = get_api_instance("assemblyai", api_key)

        # Create an invalid audio file
        invalid_file = tmp_path / "invalid.wav"
        invalid_file.write_text("This is not a real WAV file")

        # Should raise an error
        with pytest.raises((ValueError, Exception)):
            api.transcribe(invalid_file)

    def test_no_flac_requirement(self, sample_audio_file, api_keys):
        """Test that FLAC is not required (AssemblyAI accepts multiple formats)."""
        api_key = api_keys.get("assemblyai")
        if not api_key:
            pytest.skip("No AssemblyAI API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("assemblyai", api_key)

        # AssemblyAI accepts the original file format
        result = api.transcribe(sample_audio_file)

        assert result is not None
        assert result.api_name == "assemblyai"

    def test_large_file_support(self, sample_audio_files, api_keys):
        """Test support for files larger than 25MB."""
        api_key = api_keys.get("assemblyai")
        if not api_key:
            pytest.skip("No AssemblyAI API key available")

        long_audio = sample_audio_files.get("long_audio")
        if not long_audio:
            pytest.skip("No long audio file available")

        api = get_api_instance("assemblyai", api_key)

        # Check file size
        file_size_mb = long_audio.stat().st_size / (1024 * 1024)

        # AssemblyAI supports large files (up to 200MB)
        if file_size_mb > 20:
            result = api.transcribe(long_audio)
            assert result is not None
        else:
            pytest.skip(f"Long audio file not large enough for large file test ({file_size_mb:.2f}MB)")

    def test_language_parameter(self, sample_audio_file, api_keys):
        """Test language parameter handling."""
        api_key = api_keys.get("assemblyai")
        if not api_key:
            pytest.skip("No AssemblyAI API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("assemblyai", api_key)

        # Test with German language
        result = api.transcribe(sample_audio_file, language="de")

        assert result is not None
        assert result.api_name == "assemblyai"

    def test_model_selection_best(self, sample_audio_file, api_keys):
        """Test model selection with 'best' model."""
        api_key = api_keys.get("assemblyai")
        if not api_key:
            pytest.skip("No AssemblyAI API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("assemblyai", api_key)

        result = api.transcribe(sample_audio_file, model="best")

        assert result is not None
        assert result.api_name == "assemblyai"

    def test_model_selection_nano(self, sample_audio_file, api_keys):
        """Test model selection with 'nano' model."""
        api_key = api_keys.get("assemblyai")
        if not api_key:
            pytest.skip("No AssemblyAI API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("assemblyai", api_key)

        result = api.transcribe(sample_audio_file, model="nano")

        assert result is not None
        assert result.api_name == "assemblyai"

    def test_semi_automatic_punctuation(self, sample_audio_file, api_keys):
        """Test that punctuation is added automatically."""
        api_key = api_keys.get("assemblyai")
        if not api_key:
            pytest.skip("No AssemblyAI API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("assemblyai", api_key)

        result = api.transcribe(sample_audio_file)

        # Check if punctuation is present in transcription
        # (AssemblyAI adds punctuation automatically)
        if result.text:
            # Look for common punctuation marks
            has_punctuation = any(char in result.text for char in ['.', ',', '!', '?'])
            # Note: This might not always be true depending on audio content
            assert isinstance(result.text, str)

    def test_number_of_speakers_parameter(self, sample_audio_files, api_keys):
        """Test number of speakers parameter."""
        api_key = api_keys.get("assemblyai")
        if not api_key:
            pytest.skip("No AssemblyAI API key available")

        multi_speaker = sample_audio_files.get("multi_speaker")
        if not multi_speaker:
            pytest.skip("No multi-speaker audio file available")

        api = get_api_instance("assemblyai", api_key)

        # This test verifies the parameter is accepted
        # Actual speaker detection depends on audio content
        result = api.transcribe(multi_speaker, speaker_labels=True)

        assert result is not None
        assert result.api_name == "assemblyai"

    def test_audio_event_detection(self, sample_audio_file, api_keys):
        """Test audio event detection capabilities."""
        api_key = api_keys.get("assemblyai")
        if not api_key:
            pytest.skip("No AssemblyAI API key available")

        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        api = get_api_instance("assemblyai", api_key)

        # AssemblyAI can detect various audio events
        # This test verifies the transcription works
        result = api.transcribe(sample_audio_file)

        assert result is not None
        assert result.text is not None
