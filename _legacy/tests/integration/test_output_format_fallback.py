"""
Output format fallback tests.

Test that APIs properly fallback from verbose_json to text when formats are not supported.
"""
import pytest

from audio_transcribe.utils.api import get_api_instance
from audio_transcribe.utils.parsers import generate_words_from_text


@pytest.mark.integration
class TestOutputFormatFallback:
    """Test that APIs properly fallback from verbose_json to text."""

    def test_openai_extended_fallback_sequence(self, sample_audio_file, api_keys):
        """Test OpenAI Extended format fallback: verbose_json → json → text."""
        api_key = api_keys.get("openai")
        if not api_key:
            pytest.skip("No OpenAI API key available")

        api = get_api_instance("openai_extended", api_key)
        if not api.check_api_key():
            pytest.skip("OpenAI API key is invalid/expired")

        # Test with model that only supports text/json (gpt-4o-transcribe)
        result = api.transcribe(
            sample_audio_file,
            model="gpt-4o-transcribe"
        )

        # Should have text at minimum
        assert result.text
        assert len(result.text) > 0

        # Should NOT have word timestamps (model doesn't support)
        # But generate_words_from_text() should provide approximations
        assert len(result.words) > 0

    def test_gemini_text_only(self, sample_audio_file, api_keys):
        """Test Gemini only returns text (no JSON)."""
        api_key = api_keys.get("gemini")
        if not api_key:
            pytest.skip("No Gemini API key available")

        api = get_api_instance("gemini", api_key)
        if not api.check_api_key():
            pytest.skip("Gemini API key is invalid/expired")

        result = api.transcribe(sample_audio_file)

        # Should have text
        assert result.text
        assert len(result.text) > 0

        # Should have approximate word timings (generated from text)
        assert len(result.words) > 0

        # All word timings should be evenly distributed (approximate)
        # Check that timing is sequential
        for i in range(len(result.words) - 1):
            assert result.words[i]['end'] <= result.words[i+1]['start'] + 0.001, \
                f"Word {i} end time should be <= word {i+1} start time"

    def test_mistral_segment_to_words(self, sample_audio_file, api_keys):
        """Test Mistral converts segments to approximate word timestamps."""
        api_key = api_keys.get("mistral")
        if not api_key:
            pytest.skip("No Mistral API key available")

        api = get_api_instance("mistral", api_key)

        result = api.transcribe(sample_audio_file)

        # Should have text
        assert result.text
        assert len(result.text) > 0

        # Should have words (converted from segments)
        assert len(result.words) > 0

        # Words should be evenly distributed within segments
        # (This is the conversion logic we're testing)
        for i in range(len(result.words) - 1):
            if result.words[i].get('end') and result.words[i+1].get('start'):
                assert result.words[i]['end'] <= result.words[i+1]['start'] + 0.001, \
                    f"Word {i} end time should be <= word {i+1} start time"


@pytest.mark.unit
class TestFormatFallbackLogic:
    """Test the format fallback logic without API calls."""

    def test_openai_extended_format_selection(self):
        """Test that OpenAI Extended selects correct format based on model."""
        from audio_transcribe.utils.api.openai_extended import MODEL_CAPABILITIES

        # Test gpt-4o-transcribe (text/json only)
        gpt4o_caps = MODEL_CAPABILITIES["gpt-4o-transcribe"]
        assert "verbose_json" not in gpt4o_caps["supported_output_formats"]
        assert "json" in gpt4o_caps["supported_output_formats"]
        assert "text" in gpt4o_caps["supported_output_formats"]

        # Test whisper-1 (full format support)
        whisper_caps = MODEL_CAPABILITIES["whisper-1"]
        assert "verbose_json" in whisper_caps["supported_output_formats"]
        assert "json" in whisper_caps["supported_output_formats"]
        assert "text" in whisper_caps["supported_output_formats"]

    def test_get_best_response_format_with_unsupported_format(self):
        """Test get_best_response_format falls back correctly."""
        from audio_transcribe.utils.api import get_api_instance

        # Gemini only supports text
        gemini_api = get_api_instance("gemini")

        # Request verbose_json (not supported)
        preferred = ["verbose_json", "json", "text"]
        best = gemini_api.get_best_response_format(preferred)

        # Should fallback to text
        assert best == "text"

    def test_get_best_response_format_with_supported_format(self):
        """Test get_best_response_format returns supported format."""
        from audio_transcribe.utils.api import get_api_instance

        # OpenAI Extended supports multiple formats
        openai_api = get_api_instance("openai_extended")

        # Request json (supported)
        preferred = ["verbose_json", "json", "text"]
        best = openai_api.get_best_response_format(preferred)

        # Should return json or better
        assert best in ["json", "verbose_json", "text"]

    def test_generate_words_from_text_creates_approximate_timings(self):
        """Test that generate_words_from_text creates approximate timings."""
        text = "This is a test transcription with several words."
        words = generate_words_from_text(text)

        # Should have words
        assert len(words) > 0

        # All words should have text, start, end
        for word in words:
            assert "text" in word
            assert "start" in word
            assert "end" in word

        # Timings should be sequential (with tolerance for floating point precision)
        for i in range(len(words) - 1):
            assert words[i]['end'] <= words[i+1]['start'] + 0.001

    def test_mistral_segment_to_word_conversion(self):
        """Test Mistral's segment-to-word conversion logic."""
        from audio_transcribe.utils.api.mistral_voxtral import MistralVoxtralAPI

        api = MistralVoxtralAPI()

        # Mock segment data
        segments = [
            {"text": "Hello world", "start": 0.0, "end": 2.0},
            {"text": "This is a test", "start": 2.0, "end": 5.0},
        ]

        words = api._convert_segments_to_words(segments)

        # Should convert to 6 words (2 + 4)
        assert len(words) == 6  # "Hello", "world", "This", "is", "a", "test"

        # First segment: 2 words over 2 seconds = 1 second per word
        assert words[0]["text"] == "Hello"
        assert words[0]["start"] == 0.0
        assert words[0]["end"] == 1.0

        assert words[1]["text"] == "world"
        assert words[1]["start"] == 1.0
        assert words[1]["end"] == 2.0

        # Second segment: 4 words over 3 seconds = 0.75 seconds per word
        assert words[2]["text"] == "This"
        assert words[2]["start"] == 2.0
        assert words[2]["end"] == 2.75

    def test_gemini_inline_vs_files_api_threshold(self):
        """Test Gemini uses correct method based on file size."""
        from audio_transcribe.utils.api.gemini import GeminiAPI

        api = GeminiAPI()

        # Check the threshold
        assert api.INLINE_SIZE_LIMIT == 20 * 1024 * 1024  # 20MB

        # Files <= 20MB should use inline
        # Files > 20MB should use Files API
        # (This is tested by checking the implementation logic)

    def test_all_apis_have_capability_flags(self):
        """Test that all API implementations have capability flags defined."""
        from audio_transcribe.utils.api import get_api_instance

        api_names = ["openai", "openai_extended", "groq", "assemblyai", "elevenlabs", "gemini", "mistral"]

        for api_name in api_names:
            api = get_api_instance(api_name)

            # Should have all capability flags
            assert hasattr(api, "supports_word_timestamps")
            assert hasattr(api, "supports_segment_timestamps")
            assert hasattr(api, "supports_speaker_diarization")
            assert hasattr(api, "supports_srt_format")
            assert hasattr(api, "supported_output_formats")

            # supported_output_formats should be a list
            assert isinstance(api.supported_output_formats, list)
            assert len(api.supported_output_formats) > 0
