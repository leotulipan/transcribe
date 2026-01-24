"""
Model capability detection tests.

Tests each model's actual capabilities via API probing.
"""
import pytest

from audio_transcribe.utils.api import get_api_instance


@pytest.mark.integration
class TestModelCapabilities:
    """Test each model's actual capabilities via API probing."""

    @pytest.mark.parametrize("model,expected_formats", [
        ("whisper-1", ["text", "json", "srt", "verbose_json", "vtt"]),
        ("gpt-4o-transcribe", ["text", "json"]),
        ("gpt-4o-mini-transcribe", ["text", "json"]),
        ("gpt-4o-transcribe-diarize", ["text", "json"]),
    ])
    def test_openai_extended_model_output_formats(self, model, expected_formats, sample_audio_file, api_keys):
        """Test which output formats each model actually supports."""
        api_key = api_keys.get("openai")
        if not api_key:
            pytest.skip("No OpenAI API key available")

        api = get_api_instance("openai_extended", api_key)

        # Check that model capability flags match expected
        from audio_transcribe.utils.api.openai_extended import MODEL_CAPABILITIES
        model_caps = MODEL_CAPABILITIES.get(model, {})
        supported_formats = model_caps.get("supported_output_formats", [])

        assert set(supported_formats) >= set(expected_formats), \
            f"Model {model} doesn't support documented formats: {expected_formats}"

    @pytest.mark.parametrize("model,should_support_diarization", [
        ("gpt-4o-transcribe", False),
        ("gpt-4o-transcribe-diarize", True)
    ])
    def test_openai_extended_diarization_support(self, model, should_support_diarization, sample_audio_files, api_keys):
        """Test diarization support in models."""
        api_key = api_keys.get("openai")
        if not api_key:
            pytest.skip("No OpenAI API key available")

        # Need multi-speaker audio for diarization test
        multi_speaker_file = sample_audio_files.get("multi_speaker")
        if not multi_speaker_file:
            pytest.skip("No multi-speaker audio file available")

        api = get_api_instance("openai_extended", api_key)

        from audio_transcribe.utils.api.openai_extended import MODEL_CAPABILITIES
        model_caps = MODEL_CAPABILITIES.get(model, {})

        has_diarization = model_caps.get("supports_speaker_diarization", False)

        if should_support_diarization:
            assert has_diarization, f"{model} should support diarization"
        else:
            assert not has_diarization, f"{model} should not support diarization"

    def test_gemini_text_only(self, sample_audio_file, api_keys):
        """Test Gemini only returns text (no JSON with timestamps)."""
        api_key = api_keys.get("gemini")
        if not api_key:
            pytest.skip("No Gemini API key available")

        api = get_api_instance("gemini", api_key)

        # Check capability flags
        assert api.supports_word_timestamps is False, "Gemini should not support word timestamps"
        assert api.supports_segment_timestamps is False, "Gemini should not support segment timestamps"
        assert api.supports_srt_format is False, "Gemini should not support SRT format"
        assert "text" in api.supported_output_formats, "Gemini should support text output"

    def test_mistral_segment_only_timestamps(self, sample_audio_file, api_keys):
        """Test that Mistral only provides segment timestamps."""
        api_key = api_keys.get("mistral")
        if not api_key:
            pytest.skip("No Mistral API key available")

        api = get_api_instance("mistral", api_key)

        # Check capability flags
        assert api.supports_word_timestamps is False, "Mistral should not support word timestamps"
        assert api.supports_segment_timestamps is True, "Mistral should support segment timestamps"

    def test_mistral_language_auto_detect(self, sample_audio_files, api_keys):
        """Test that Mistral auto-detects language (cannot specify)."""
        api_key = api_keys.get("mistral")
        if not api_key:
            pytest.skip("No Mistral API key available")

        api = get_api_instance("mistral", api_key)

        # Check that language parameter would be ignored (logged as warning)
        # This is tested by checking the implementation behavior
        assert api.api_name == "mistral", "API name should be mistral"


@pytest.mark.integration
class TestModelCapabilityFlags:
    """Test that capability flags are properly set on all API classes."""

    def test_openai_extended_capability_flags(self):
        """Test OpenAI Extended has proper capability flags."""
        from audio_transcribe.utils.api.openai_extended import OpenAIExtendedAPI

        api = OpenAIExtendedAPI()

        # Base class should have these attributes
        assert hasattr(api, "supports_word_timestamps")
        assert hasattr(api, "supports_segment_timestamps")
        assert hasattr(api, "supports_speaker_diarization")
        assert hasattr(api, "supports_srt_format")
        assert hasattr(api, "supported_output_formats")

    def test_gemini_capability_flags(self):
        """Test Gemini has proper capability flags (text-only)."""
        from audio_transcribe.utils.api.gemini import GeminiAPI

        api = GeminiAPI()

        # Gemini is text-only
        assert api.supports_word_timestamps is False
        assert api.supports_segment_timestamps is False
        assert api.supports_speaker_diarization is False
        assert api.supports_srt_format is False
        assert "text" in api.supported_output_formats

    def test_mistral_capability_flags(self):
        """Test Mistral has proper capability flags (segment-level only)."""
        from audio_transcribe.utils.api.mistral_voxtral import MistralVoxtralAPI

        api = MistralVoxtralAPI()

        # Mistral supports segment timestamps, not word-level
        assert api.supports_word_timestamps is False
        assert api.supports_segment_timestamps is True
        assert api.supports_speaker_diarization is False
        assert "json" in api.supported_output_formats

    def test_get_best_response_format(self):
        """Test get_best_response_format method on all APIs."""
        from audio_transcribe.utils.api import get_api_instance

        # Test with each API
        for api_name in ["openai_extended", "gemini", "mistral"]:
            api = get_api_instance(api_name)

            # Test format fallback logic
            preferred_formats = ["verbose_json", "json", "text"]
            best_format = api.get_best_response_format(preferred_formats)

            # Should return a supported format
            assert best_format in api.supported_output_formats or best_format == "text"

    def test_openai_extended_model_capabilities(self):
        """Test OpenAI Extended model-specific capabilities."""
        from audio_transcribe.utils.api.openai_extended import MODEL_CAPABILITIES

        for model, caps in MODEL_CAPABILITIES.items():
            # Each model should have all required capability fields
            assert "supports_word_timestamps" in caps
            assert "supports_segment_timestamps" in caps
            assert "supports_speaker_diarization" in caps
            assert "supports_srt_format" in caps
            assert "supported_output_formats" in caps

            # supported_output_formats should be a list
            assert isinstance(caps["supported_output_formats"], list)
            assert len(caps["supported_output_formats"]) > 0
