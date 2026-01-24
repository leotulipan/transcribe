"""
Unit tests for language utilities.

Tests language code mapping, language support, validation,
edge cases, and language map completeness.
"""
import pytest

from audio_transcribe.transcribe_helpers.language_utils import (
    LANGUAGE_MAP,
    get_language_code,
    get_supported_languages,
    is_language_supported,
)


@pytest.mark.unit
class TestLanguageCodeMapping:
    """Test language code conversion and mapping."""

    def test_iso639_1_to_assemblyai(self):
        """Test converting ISO-639-1 code to AssemblyAI format."""
        result = get_language_code("en", "assemblyai")
        assert result == "en"

    def test_iso639_1_to_groq(self):
        """Test converting ISO-639-1 code to Groq format."""
        result = get_language_code("de", "groq")
        assert result == "de"

    def test_iso639_1_to_openai(self):
        """Test converting ISO-639-1 code to OpenAI format."""
        result = get_language_code("fr", "openai")
        assert result == "fr"

    def test_iso639_3_to_api_code(self):
        """Test converting ISO-639-3 code to API-specific format."""
        result = get_language_code("eng", "groq")
        assert result == "en"

        result = get_language_code("deu", "assemblyai")
        assert result == "de"

    def test_language_name_to_code(self):
        """Test converting language name to code."""
        result = get_language_code("english", "openai")
        assert result == "en"

        result = get_language_code("german", "groq")
        assert result == "de"

    def test_native_name_to_code(self):
        """Test converting native language name to code."""
        result = get_language_code("Deutsch", "elevenlabs")
        assert result == "de"

        result = get_language_code("Español", "openai")
        assert result == "es"

    def test_case_insensitive_matching(self):
        """Test that language matching is case-insensitive."""
        assert get_language_code("EN", "groq") == "en"
        assert get_language_code("English", "openai") == "en"
        assert get_language_code("ENGLISH", "assemblyai") == "en"

    def test_whitespace_handling(self):
        """Test that whitespace is handled correctly."""
        assert get_language_code(" en ", "groq") == "en"
        assert get_language_code("  de  ", "openai") == "de"

    def test_api_name_case_insensitive(self):
        """Test that API name matching is case-insensitive."""
        assert get_language_code("en", "GROQ") == "en"
        assert get_language_code("de", "OpenAI") == "de"

    def test_fallback_to_iso639_1(self):
        """Test fallback to ISO-639-1 when API-specific code not found."""
        # All current languages have API codes, but test the logic
        result = get_language_code("en", "unknown_api")
        assert result == "en"

    def test_unknown_language_returns_as_is(self):
        """Test that unknown language codes are returned as-is."""
        result = get_language_code("xx", "groq")
        assert result == "xx"

    def test_empty_input_returns_none(self):
        """Test that empty input returns None."""
        assert get_language_code("", "groq") is None
        assert get_language_code(None, "openai") is None


@pytest.mark.unit
class TestLanguageSupport:
    """Test language support checking."""

    def test_check_supported_language(self):
        """Test checking if a language is supported."""
        assert is_language_supported("en")
        assert is_language_supported("de")
        assert is_language_supported("fr")

    def test_check_unsupported_language(self):
        """Test checking unsupported language."""
        assert not is_language_supported("xx")
        assert not is_language_supported("unknown")

    def test_support_by_iso639_3(self):
        """Test language support by ISO-639-3 code."""
        assert is_language_supported("eng")
        assert is_language_supported("deu")
        assert is_language_supported("fra")

    def test_support_by_name(self):
        """Test language support by name."""
        assert is_language_supported("english")
        assert is_language_supported("german")
        assert is_language_supported("french")

    def test_support_by_native_name(self):
        """Test language support by native name."""
        assert is_language_supported("Deutsch")
        assert is_language_supported("Español")
        assert is_language_supported("Français")


@pytest.mark.unit
class TestLanguageMapCompleteness:
    """Test completeness and consistency of language map."""

    def test_all_languages_have_iso639_1(self):
        """Test that all languages have ISO-639-1 codes."""
        for code, data in LANGUAGE_MAP.items():
            if code not in ['deu', 'eng', 'fra', 'spa']:  # Aliases
                assert 'iso639_1' in data, f"{code} missing iso639_1"

    def test_all_languages_have_iso639_3(self):
        """Test that all languages have ISO-639-3 codes."""
        for code, data in LANGUAGE_MAP.items():
            assert 'iso639_3' in data, f"{code} missing iso639_3"

    def test_all_languages_have_names(self):
        """Test that all languages have both name and native_name."""
        for code, data in LANGUAGE_MAP.items():
            assert 'name' in data, f"{code} missing name"
            assert 'native_name' in data, f"{code} missing native_name"

    def test_supported_apis_consistent(self):
        """Test that all languages support the same set of APIs."""
        expected_apis = {'assemblyai', 'elevenlabs', 'groq', 'openai'}

        for code, data in LANGUAGE_MAP.items():
            # Check that expected APIs are present
            for api in expected_apis:
                assert api in data, f"{code} missing {api} code"

    def test_no_duplicate_primary_codes(self):
        """Test that there are no duplicate primary language codes."""
        primary_codes = [k for k in LANGUAGE_MAP.keys() if len(k) == 2]
        assert len(primary_codes) == len(set(primary_codes)), "Duplicate primary codes found"


@pytest.mark.unit
class TestGetSupportedLanguages:
    """Test get_supported_languages function."""

    def test_returns_non_empty_set(self):
        """Test that function returns non-empty set."""
        supported = get_supported_languages()
        assert len(supported) > 0

    def test_includes_iso639_1_codes(self):
        """Test that supported languages include ISO-639-1 codes."""
        supported = get_supported_languages()
        assert 'en' in supported
        assert 'de' in supported
        assert 'fr' in supported

    def test_includes_iso639_3_codes(self):
        """Test that supported languages include ISO-639-3 codes."""
        supported = get_supported_languages()
        assert 'eng' in supported
        assert 'deu' in supported
        assert 'fra' in supported

    def test_includes_language_names(self):
        """Test that supported languages include language names."""
        supported = get_supported_languages()
        assert 'english' in supported
        assert 'german' in supported
        assert 'french' in supported

    def test_includes_native_names(self):
        """Test that supported languages include native names (lowercase)."""
        supported = get_supported_languages()
        # Native names are stored in lowercase in the set
        assert 'deutsch' in supported
        assert 'español' in supported

    def test_returns_set_type(self):
        """Test that function returns a set."""
        supported = get_supported_languages()
        assert isinstance(supported, set)


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_special_characters_in_name(self):
        """Test handling of special characters in native names."""
        # Japanese and Chinese have non-Latin characters
        result = get_language_code("日本語", "groq")
        assert result == "ja"

        result = get_language_code("中文", "openai")
        assert result == "zh"

    def test_variants_and_aliases(self):
        """Test that language variants and aliases work."""
        # deu should map to de
        assert get_language_code("deu", "groq") == "de"

        # german should map to de
        assert get_language_code("german", "openai") == "de"

        # eng should map to en
        assert get_language_code("eng", "assemblyai") == "en"

    def test_multiple_ways_to_specify_same_language(self):
        """Test that same language can be specified multiple ways."""
        results = [
            get_language_code("en", "groq"),
            get_language_code("eng", "groq"),
            get_language_code("english", "groq"),
            get_language_code("English", "groq"),
        ]
        assert all(r == "en" for r in results)

    def test_empty_language_input(self):
        """Test handling of empty language input."""
        assert is_language_supported("") is False
        assert is_language_supported(None) is False

    def test_whitespace_only_input(self):
        """Test handling of whitespace-only input."""
        assert is_language_supported("   ") is False
        # Whitespace-only input returns empty string, not None
        result = get_language_code("   ", "groq")
        assert result == "" or result is None

    def test_very_long_language_code(self):
        """Test handling of unrealistically long language code."""
        result = get_language_code("verylonglanguagecode", "groq")
        # Should return as-is with a warning
        assert result == "verylonglanguagecode"

    def test_numeric_language_code(self):
        """Test handling of numeric language code."""
        result = get_language_code("123", "groq")
        # Should return as-is
        assert result == "123"

    def test_mixed_case_api_name(self):
        """Test handling of mixed case API name."""
        result = get_language_code("en", "GrOq")
        assert result == "en"

        result = get_language_code("de", "OpEnAi")
        assert result == "de"


@pytest.mark.unit
class TestAsianLanguages:
    """Test Asian language support specifically."""

    def test_japanese_support(self):
        """Test Japanese language support."""
        assert is_language_supported("ja")
        assert is_language_supported("jpn")
        assert is_language_supported("日本語")
        assert get_language_code("ja", "groq") == "ja"

    def test_chinese_support(self):
        """Test Chinese language support."""
        assert is_language_supported("zh")
        assert is_language_supported("zho")
        assert is_language_supported("中文")
        assert get_language_code("zh", "openai") == "zh"

    def test_korean_support(self):
        """Test Korean language support."""
        assert is_language_supported("ko")
        assert is_language_supported("kor")
        assert is_language_supported("한국어")
        assert get_language_code("ko", "assemblyai") == "ko"


@pytest.mark.unit
class TestEuropeanLanguages:
    """Test European language support."""

    def test_italian_support(self):
        """Test Italian language support."""
        assert is_language_supported("it")
        assert get_language_code("it", "groq") == "it"

    def test_dutch_support(self):
        """Test Dutch language support."""
        assert is_language_supported("nl")
        assert get_language_code("nl", "openai") == "nl"

    def test_polish_support(self):
        """Test Polish language support."""
        assert is_language_supported("pl")
        assert get_language_code("pl", "assemblyai") == "pl"

    def test_portuguese_support(self):
        """Test Portuguese language support."""
        assert is_language_supported("pt")
        assert get_language_code("pt", "elevenlabs") == "pt"

    def test_russian_support(self):
        """Test Russian language support."""
        assert is_language_supported("ru")
        assert get_language_code("ru", "groq") == "ru"

    def test_swedish_support(self):
        """Test Swedish language support."""
        assert is_language_supported("sv")
        assert get_language_code("sv", "openai") == "sv"


@pytest.mark.unit
class TestAPIConsistency:
    """Test consistency across different APIs."""

    def test_all_apis_support_english(self):
        """Test that all APIs support English."""
        apis = ['assemblyai', 'elevenlabs', 'groq', 'openai']
        for api in apis:
            result = get_language_code("en", api)
            assert result == "en", f"{api} doesn't support en properly"

    def test_all_apis_support_german(self):
        """Test that all APIs support German."""
        apis = ['assemblyai', 'elevenlabs', 'groq', 'openai']
        for api in apis:
            result = get_language_code("de", api)
            assert result == "de", f"{api} doesn't support de properly"

    def test_all_apis_support_spanish(self):
        """Test that all APIs support Spanish."""
        apis = ['assemblyai', 'elevenlabs', 'groq', 'openai']
        for api in apis:
            result = get_language_code("es", api)
            assert result == "es", f"{api} doesn't support es properly"

    def test_all_apis_support_japanese(self):
        """Test that all APIs support Japanese."""
        apis = ['assemblyai', 'elevenlabs', 'groq', 'openai']
        for api in apis:
            result = get_language_code("ja", api)
            assert result == "ja", f"{api} doesn't support ja properly"
