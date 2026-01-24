"""
Unit tests for output formatters.

Tests the formatting logic for different output formats (text, SRT, etc.).
"""
import pytest
from pathlib import Path

from audio_transcribe.utils.formatters import (
    create_text_file,
    create_srt_file,
)
from audio_transcribe.utils.parsers import TranscriptionResult


@pytest.mark.unit
class TestCreateTextFile:
    """Test text file formatter."""

    def test_create_text_from_result(self, tmp_path):
        """Test creating text file from TranscriptionResult."""
        result = TranscriptionResult(
            text="Hello world, this is a test.",
            api_name="test"
        )

        output_file = tmp_path / "test.txt"
        create_text_file(result, output_file)

        assert output_file.exists()

        content = output_file.read_text(encoding='utf-8')
        assert "Hello world, this is a test." in content

    def test_create_text_from_words_only(self, tmp_path):
        """Test creating text file when only words are available."""
        result = TranscriptionResult(
            text="",  # Empty text
            words=[
                {"text": "Hello", "start": 0.0, "end": 0.5},
                {"text": "world", "start": 0.5, "end": 1.0},
            ],
            api_name="test"
        )

        output_file = tmp_path / "test.txt"
        create_text_file(result, output_file)

        content = output_file.read_text(encoding='utf-8')
        assert "Hello" in content
        assert "world" in content

    def test_create_text_handles_empty_result(self, tmp_path):
        """Test creating text file with empty result."""
        result = TranscriptionResult(
            text="",
            words=[],
            api_name="test"
        )

        output_file = tmp_path / "test.txt"
        create_text_file(result, output_file)

        content = output_file.read_text(encoding='utf-8')
        assert content == ""


@pytest.mark.unit
class TestCreateSRTFile:
    """Test SRT file formatter."""

    def test_create_standard_srt(self, tmp_path):
        """Test creating standard SRT file."""
        result = TranscriptionResult(
            text="Hello world",
            words=[
                {"text": "Hello", "start": 0.0, "end": 1.0},
                {"text": "world", "start": 1.0, "end": 2.0},
            ],
            api_name="test"
        )

        output_file = tmp_path / "test.srt"
        create_srt_file(result, output_file, format_type="standard")

        assert output_file.exists()

        content = output_file.read_text(encoding='utf-8')

        # Should have SRT format markers
        assert " --> " in content  # Time separator
        assert "1\n" in content or "1\r\n" in content  # Subtitle number

    def test_create_word_srt(self, tmp_path):
        """Test creating word-level SRT file."""
        result = TranscriptionResult(
            text="Hello world",
            words=[
                {"text": "Hello", "start": 0.0, "end": 0.5},
                {"text": "world", "start": 0.5, "end": 1.0},
            ],
            api_name="test"
        )

        output_file = tmp_path / "test.word.srt"
        create_srt_file(result, output_file, format_type="word")

        assert output_file.exists()

        content = output_file.read_text(encoding='utf-8')

        # Word SRT should have each word as a subtitle
        assert "Hello" in content
        assert "world" in content

    def test_create_davinci_srt(self, tmp_path):
        """Test creating DaVinci Resolve SRT file."""
        result = TranscriptionResult(
            text="Hello world",
            words=[
                {"text": "Hello", "start": 0.0, "end": 0.5},
                {"text": "world", "start": 0.5, "end": 1.0},
            ],
            api_name="test"
        )

        output_file = tmp_path / "test.davinci.srt"
        create_srt_file(
            result,
            output_file,
            format_type="davinci",
            silent_portions=500,
            filler_lines=False
        )

        assert output_file.exists()

        content = output_file.read_text(encoding='utf-8')

        # DaVinci SRT should have timing information
        assert " --> " in content

    def test_create_srt_with_empty_words(self, tmp_path):
        """Test creating SRT file with empty words list."""
        result = TranscriptionResult(
            text="Hello world",
            words=[],
            api_name="test"
        )

        output_file = tmp_path / "test.srt"

        # Should handle gracefully
        create_srt_file(result, output_file, format_type="standard")

        # File should exist even if empty
        assert output_file.exists()


@pytest.mark.unit
class TestSRTTimestampFormatting:
    """Test SRT timestamp formatting utilities."""

    def test_timestamp_conversion(self):
        """Test conversion from seconds to SRT timestamp format."""
        from audio_transcribe.transcribe_helpers.output_formatters import format_time, format_time_ms

        # Test format_time (seconds as float)
        assert format_time(0.0) == "00:00:00,000"
        assert format_time(1.0) == "00:00:01,000"
        assert format_time(65.0) == "00:01:05,000"
        assert format_time(3661.5) == "01:01:01,500"

        # Test format_time_ms (milliseconds as int)
        assert format_time_ms(0) == "00:00:00,000"
        assert format_time_ms(1000) == "00:00:01,000"
        assert format_time_ms(65000) == "00:01:05,000"
        assert format_time_ms(3661500) == "01:01:01,500"

    def test_timestamp_boundary_cases(self):
        """Test timestamp formatting at boundaries."""
        from audio_transcribe.transcribe_helpers.output_formatters import format_time_ms

        # Test hour rollover
        assert format_time_ms(3600000) == "01:00:00,000"
        assert format_time_ms(7200000) == "02:00:00,000"


@pytest.mark.unit
class TestCreateOutputFiles:
    """Test the main create_output_files function."""

    def test_create_text_and_srt(self, tmp_path):
        """Test creating multiple output formats at once."""
        from audio_transcribe.utils.formatters import create_output_files

        result = TranscriptionResult(
            text="Hello world",
            words=[
                {"text": "Hello", "start": 0.0, "end": 1.0},
                {"text": "world", "start": 1.0, "end": 2.0},
            ],
            api_name="test"
        )

        output_path = tmp_path / "output"
        output_formats = ["text", "srt"]

        create_output_files(
            result,
            output_path,
            output_formats=output_formats,
            **{
                "filler_words": ["um", "uh"],
                "silent_portions": 500,
            }
        )

        # Check files were created
        assert (tmp_path / "output.txt").exists()
        assert (tmp_path / "output.srt").exists()


@pytest.mark.unit
class TestFormatterEdgeCases:
    """Test edge cases in formatters."""

    def test_empty_result(self, tmp_path):
        """Test formatting completely empty result."""
        result = TranscriptionResult(
            text="",
            words=[],
            api_name="test"
        )

        output_file = tmp_path / "empty.txt"
        create_text_file(result, output_file)

        content = output_file.read_text(encoding='utf-8')
        assert content == ""

    def test_unicode_characters(self, tmp_path):
        """Test formatting text with unicode characters."""
        result = TranscriptionResult(
            text="Hello 世界 🌍",
            words=[
                {"text": "Hello", "start": 0.0, "end": 1.0},
                {"text": "世界", "start": 1.0, "end": 2.0},
                {"text": "🌍", "start": 2.0, "end": 3.0},
            ],
            api_name="test"
        )

        output_file = tmp_path / "unicode.txt"
        create_text_file(result, output_file)

        content = output_file.read_text(encoding='utf-8')
        assert "世界" in content
        assert "🌍" in content

    def test_very_long_text(self, tmp_path):
        """Test formatting very long transcription."""
        # Generate a long transcription
        words = []
        text_parts = []
        for i in range(1000):
            word_text = f"word{i}"
            words.append({
                "text": word_text,
                "start": float(i),
                "end": float(i + 1)
            })
            text_parts.append(word_text)

        result = TranscriptionResult(
            text=" ".join(text_parts),
            words=words,
            api_name="test"
        )

        output_file = tmp_path / "long.txt"
        create_text_file(result, output_file)

        # Should successfully write long file
        assert output_file.exists()
        content = output_file.read_text(encoding='utf-8')
        assert len(content) > 1000

    def test_words_with_missing_fields(self, tmp_path):
        """Test handling words with missing optional fields."""
        result = TranscriptionResult(
            text="Hello world",
            words=[
                {"text": "Hello"},  # Missing start/end
                {"text": "world", "start": 0.5, "end": 1.0},
            ],
            api_name="test"
        )

        output_file = tmp_path / "test.txt"
        create_text_file(result, output_file)

        # Should handle gracefully
        assert output_file.exists()
        content = output_file.read_text(encoding='utf-8')
        assert "Hello" in content or "world" in content
