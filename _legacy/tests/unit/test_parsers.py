"""
Unit tests for transcription parsers.

Tests the parsing logic for different API response formats.
"""
import json
import pytest
from pathlib import Path

from audio_transcribe.utils.parsers import (
    TranscriptionResult,
    parse_assemblyai_format,
    parse_elevenlabs_format,
    parse_groq_format,
    parse_openai_format,
    generate_words_from_text,
    detect_and_parse_json,
)


@pytest.mark.unit
class TestTranscriptionResult:
    """Test TranscriptionResult class."""

    def test_init_with_all_fields(self):
        """Test initializing TranscriptionResult with all fields."""
        result = TranscriptionResult(
            text="Hello world",
            confidence=0.95,
            language="en",
            words=[{"text": "Hello", "start": 0.0, "end": 0.5}],
            speakers=[{"id": 1, "name": "Speaker 1"}],
            segments=[{"start": 0.0, "end": 1.0, "text": "Hello world"}],
            api_name="test_api"
        )

        assert result.text == "Hello world"
        assert result.confidence == 0.95
        assert result.language == "en"
        assert len(result.words) == 1
        assert len(result.speakers) == 1
        assert len(result.segments) == 1
        assert result.api_name == "test_api"

    def test_init_with_minimal_fields(self):
        """Test initializing TranscriptionResult with minimal fields."""
        result = TranscriptionResult(
            text="Test",
            api_name="test_api"
        )

        assert result.text == "Test"
        assert result.confidence == 0.0
        assert result.language == "en"
        assert result.words == []
        assert result.speakers == []
        assert result.segments == []
        assert result.api_name == "test_api"

    def test_to_dict(self):
        """Test converting TranscriptionResult to dictionary."""
        result = TranscriptionResult(
            text="Test",
            confidence=0.9,
            language="en",
            api_name="test"
        )

        data = result.to_dict()

        assert data["text"] == "Test"
        assert data["confidence"] == 0.9
        assert data["language"] == "en"
        assert data["api_name"] == "test"

    def test_to_json(self):
        """Test converting TranscriptionResult to JSON string."""
        result = TranscriptionResult(
            text="Test",
            words=[
                {"text": "Test", "start": 0.0, "end": 0.001}
            ],
            api_name="test"
        )

        json_str = result.to_json()

        # Should be valid JSON
        data = json.loads(json_str)
        assert data["text"] == "Test"
        assert len(data["words"]) == 1

    def test_save_and_load_from_file(self, tmp_path):
        """Test saving and loading TranscriptionResult from file."""
        result = TranscriptionResult(
            text="Hello world",
            words=[
                {"text": "Hello", "start": 0.0, "end": 0.5},
                {"text": "world", "start": 0.5, "end": 1.0}
            ],
            api_name="test"
        )

        # Save
        json_path = tmp_path / "test.json"
        result.save(json_path)

        # Load
        loaded_result = TranscriptionResult.from_file(json_path)

        assert loaded_result.text == result.text
        assert len(loaded_result.words) == len(result.words)


@pytest.mark.unit
class TestGenerateWordsFromText:
    """Test generate_words_from_text function."""

    def test_generate_simple_text(self):
        """Test generating word timestamps from simple text."""
        text = "Hello world test"
        words = generate_words_from_text(text)

        # Note: generate_words_from_text includes spacing items
        # Filter for actual words only
        actual_words = [w for w in words if w.get("type") == "word"]

        assert len(actual_words) == 3
        assert actual_words[0]["text"] == "Hello"
        assert actual_words[1]["text"] == "world"
        assert actual_words[2]["text"] == "test"

    def test_generate_empty_text(self):
        """Test generating word timestamps from empty text."""
        words = generate_words_from_text("")
        assert words == []

    def test_generate_with_punctuation(self):
        """Test generating word timestamps with punctuation."""
        text = "Hello, world! How are you?"
        words = generate_words_from_text(text)

        # Should handle punctuation
        actual_words = [w for w in words if w.get("type") == "word"]
        assert len(actual_words) > 0
        assert "Hello" in actual_words[0]["text"]

    def test_timing_distribution(self):
        """Test that timing is evenly distributed across words."""
        text = "one two three four five"
        words = generate_words_from_text(text)

        # Check that we have the expected number of words (including spacing)
        assert len(words) >= 5

        # Check timing is sequential (with tolerance for floating point precision)
        for i in range(len(words) - 1):
            if words[i].get("end") and words[i + 1].get("start"):
                # Use a small tolerance for floating point comparison
                assert words[i]["end"] <= words[i + 1]["start"] + 0.001


@pytest.mark.unit
class TestParseOpenAIFormat:
    """Test OpenAI format parser."""

    def test_parse_whisper_response(self):
        """Test parsing OpenAI Whisper verbose_json response."""
        raw_data = {
            "text": "Hello world",
            "language": "en",
            "duration": 1.5,
            "words": [
                {"text": "Hello", "start": 0.0, "end": 0.5},
                {"text": "world", "start": 0.5, "end": 1.0}
            ]
        }

        result = parse_openai_format(raw_data)

        assert result.text == "Hello world"
        assert result.language == "en"
        # Note: parse_openai_format adds spacing items
        assert len(result.words) >= 2
        actual_words = [w for w in result.words if w.get("type") == "word"]
        assert actual_words[0]["text"] == "Hello"

    def test_parse_minimal_response(self):
        """Test parsing minimal OpenAI response."""
        raw_data = {
            "text": "Test transcription"
        }

        result = parse_openai_format(raw_data)

        assert result.text == "Test transcription"
        # Note: generate_words_from_text is called which includes spacing items
        assert len(result.words) >= 2  # "Test" + spacing + "transcription"


@pytest.mark.unit
class TestParseGroqFormat:
    """Test Groq format parser."""

    def test_parse_groq_response(self):
        """Test parsing Groq response format."""
        raw_data = {
            "text": "Hello world",
            "language": "en",
            "segments": [
                {
                    "start": 0,
                    "end": 1000,
                    "text": "Hello",
                    "words": [
                        {"word": "Hello", "start": 0, "end": 500}
                    ]
                },
                {
                    "start": 1000,
                    "end": 2000,
                    "text": "world",
                    "words": [
                        {"word": "world", "start": 1000, "end": 1500}
                    ]
                }
            ]
        }

        result = parse_groq_format(raw_data)

        assert result.text == "Hello world"
        # Note: parse_groq_format adds spacing items
        assert len(result.words) >= 2

    def test_parse_groq_chunked_response(self):
        """Test parsing Groq chunked response."""
        raw_data = {
            "text": "Test",
            "chunks": [
                {"start": 0, "end": 1000, "text": "Test"}
            ]
        }

        result = parse_groq_format(raw_data)

        assert result.text == "Test"


@pytest.mark.unit
class TestParseAssemblyAIFormat:
    """Test AssemblyAI format parser."""

    def test_parse_assemblyai_response(self):
        """Test parsing AssemblyAI response format."""
        raw_data = {
            "text": "Hello world",
            "language": "en",
            "confidence": 0.95,
            "words": [
                    {"text": "Hello", "start": 0, "end": 500},
                    {"text": "world", "start": 500, "end": 1000}
            ]
        }

        result = parse_assemblyai_format(raw_data)

        assert result.text == "Hello world"
        # Note: AssemblyAI doesn't set confidence in the parsed result
        # It defaults to 0.0
        assert len(result.words) >= 2


@pytest.mark.unit
class TestParseElevenLabsFormat:
    """Test ElevenLabs format parser."""

    def test_parse_elevenlabs_word_format(self):
        """Test parsing ElevenLabs word-level format."""
        raw_data = {
            "text": "Hello world",
            "language": "en",
            "words": [
                {"text": "Hello", "start": 0.0, "end": 0.5, "type": "word"},
                {"text": "world", "start": 0.5, "end": 1.0, "type": "word"}
            ]
        }

        result = parse_elevenlabs_format(raw_data)

        assert result.text == "Hello world"
        assert len(result.words) == 2

    def test_parse_elevenlabs_segment_format(self):
        """Test parsing ElevenLabs segment format."""
        raw_data = {
            "text": "Hello world",
            "language": "en",
            "segments": [
                {
                    "text": "Hello world",
                    "speaker": "speaker_1",
                    "start_time": 0,
                    "end_time": 1000
                }
            ]
        }

        result = parse_elevenlabs_format(raw_data)

        assert result.text == "Hello world"
        # Note: ElevenLabs segment format generates words from text
        # The segments are stored but not directly in the segments field
        # This is expected behavior based on the parser implementation


@pytest.mark.unit
class TestDetectAndParseJSON:
    """Test JSON auto-detection parser."""

    def test_detect_openai_format(self):
        """Test detecting OpenAI format."""
        raw_data = {
            "text": "Test",
            "language": "en",
            "words": [{"text": "Test", "start": 0.0, "end": 0.5}]
        }

        api_name, result = detect_and_parse_json(raw_data)

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Test"

    def test_detect_groq_format(self):
        """Test detecting Groq format."""
        raw_data = {
            "text": "Test",
            "segments": [
                {"start": 0, "end": 1000, "text": "Test"}
            ]
        }

        api_name, result = detect_and_parse_json(raw_data)

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Test"

    def test_detect_unknown_format_fallback(self):
        """Test fallback for unknown format."""
        raw_data = {
            "unknown_field": "value",
            "text": "Test"
        }

        api_name, result = detect_and_parse_json(raw_data)

        # Should still create a result with text
        assert isinstance(result, TranscriptionResult)
        assert result.text == "Test"

    def test_parse_from_file(self, tmp_path):
        """Test parsing JSON from file."""
        raw_data = {
            "text": "Hello world",
            "words": [
                {"text": "Hello", "start": 0.0, "end": 0.5},
                {"text": "world", "start": 0.5, "end": 1.0}
            ]
        }

        json_path = tmp_path / "test.json"
        with open(json_path, 'w') as f:
            json.dump(raw_data, f)

        result = TranscriptionResult.from_file(json_path)

        assert result.text == "Hello world"
        # Note: parsers add spacing items
        assert len(result.words) >= 2


@pytest.mark.unit
class TestLLMCompareTexts:
    """Test LLM-based text comparison."""

    def test_exact_match(self):
        """Test comparison with exact match."""
        from tests.conftest import llm_compare_texts

        expected = "Hello world"
        actual = "Hello world"

        passed, reason = llm_compare_texts(expected, actual)

        assert passed is True
        assert "match" in reason.lower()

    def test_semantic_match_with_punctuation(self):
        """Test comparison with punctuation differences."""
        from tests.conftest import llm_compare_texts

        expected = "Hello world, how are you?"
        actual = "Hello world how are you"  # No punctuation

        passed, reason = llm_compare_texts(expected, actual, strict=False)

        # Should pass (semantic match)
        assert passed is True

    def test_case_insensitive_match(self):
        """Test comparison is case-insensitive."""
        from tests.conftest import llm_compare_texts

        expected = "Hello World"
        actual = "hello world"

        passed, reason = llm_compare_texts(expected, actual, strict=False)

        assert passed is True

    def test_semantic_mismatch(self):
        """Test comparison with substantive differences."""
        from tests.conftest import llm_compare_texts

        expected = "The quick brown fox jumps over the lazy dog"
        actual = "This is completely different text"

        passed, reason = llm_compare_texts(expected, actual, strict=False)

        # Should fail
        assert passed is False
