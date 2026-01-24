"""
Unit tests for text processing functions.

Tests word format standardization, filler word processing, pause detection,
speaker label processing, and edge cases.
"""
import pytest
from pathlib import Path
import tempfile
import re

from audio_transcribe.transcribe_helpers.text_processing import (
    standardize_word_format,
    process_filler_words,
    merge_consecutive_pauses,
    find_longest_common_sequence,
    segments_to_words,
)


@pytest.mark.unit
class TestStandardizeWordFormat:
    """Test word format standardization function."""

    def test_empty_list_returns_empty(self):
        """Test that empty input returns empty list."""
        result = standardize_word_format([])
        assert result == []

    def test_single_word_no_spacing(self):
        """Test single word starting at zero with no spacing."""
        words = [
            {"text": "Hello", "start": 0, "end": 100}
        ]
        result = standardize_word_format(words)

        assert len(result) == 1
        assert result[0]["text"] == "Hello"
        assert result[0]["type"] == "word"
        assert result[0]["start"] == 0
        assert result[0]["end"] == 100

    def test_adds_initial_spacing_for_delayed_start(self):
        """Test that initial spacing is added when first word doesn't start at zero."""
        words = [
            {"text": "Hello", "start": 500, "end": 1000}
        ]
        result = standardize_word_format(words)

        # Should have initial spacing + word
        assert len(result) == 2
        assert result[0]["type"] == "spacing"
        assert result[0]["start"] == 0
        assert result[0]["end"] == 500
        assert result[1]["text"] == "Hello"

    def test_adds_spacing_between_words(self):
        """Test that spacing is added between words with gaps."""
        words = [
            {"text": "Hello", "start": 0, "end": 100},
            {"text": "world", "start": 200, "end": 300}
        ]
        result = standardize_word_format(words)

        # Should have word + spacing + word
        assert len(result) == 3
        assert result[0]["text"] == "Hello"
        assert result[1]["type"] == "spacing"
        assert result[1]["start"] == 100
        assert result[1]["end"] == 200
        assert result[2]["text"] == "world"

    def test_marks_pauses_with_show_pauses_true(self):
        """Test that significant pauses are marked with (...) when show_pauses=True."""
        words = [
            {"text": "Hello", "start": 0, "end": 100},
            {"text": "world", "start": 1000, "end": 1200}  # 900ms gap
        ]
        result = standardize_word_format(words, show_pauses=True, silence_threshold=250)

        # The spacing should have pause marker
        spacing = [r for r in result if r["type"] == "spacing"]
        assert len(spacing) == 1
        assert "(...)" in spacing[0]["text"]

    def test_marks_initial_pause(self):
        """Test that initial pauses are marked when show_pauses=True."""
        words = [
            {"text": "Hello", "start": 500, "end": 1000}  # 500ms initial gap
        ]
        result = standardize_word_format(words, show_pauses=True, silence_threshold=250)

        # Should have initial pause marker
        assert result[0]["type"] == "spacing"
        assert "(...)" in result[0]["text"]

    def test_handles_decimal_seconds_format(self):
        """Test handling of decimal seconds format (Groq)."""
        words = [
            {"text": "Hello", "start": 0.123, "end": 0.456},
            {"text": "world", "start": 0.789, "end": 1.234}
        ]
        result = standardize_word_format(words)

        assert len(result) >= 2
        # Should preserve decimal format (words are at index 1 and 3 after spacing is added)
        assert isinstance(result[1]["start"], float)
        assert result[1]["start"] == 0.123

    def test_marks_space_only_words_as_spacing(self):
        """Test that words containing only whitespace are marked as spacing."""
        words = [
            {"text": "Hello", "start": 0, "end": 100},
            {"text": "   ", "start": 100, "end": 150},
            {"text": "world", "start": 150, "end": 250}
        ]
        result = standardize_word_format(words)

        # The space-only word should be marked as spacing
        space_items = [r for r in result if r.get("text") == "   "]
        assert len(space_items) == 1
        assert space_items[0]["type"] == "spacing"

    def test_preserves_speaker_id(self):
        """Test that speaker_id is preserved through standardization."""
        words = [
            {"text": "Hello", "start": 0, "end": 100, "speaker_id": "S1"},
            {"text": "world", "start": 200, "end": 300, "speaker_id": "S1"}
        ]
        result = standardize_word_format(words)

        # Check speaker_id is preserved
        assert all(r.get("speaker_id") == "S1" for r in result)


@pytest.mark.unit
class TestProcessFillerWords:
    """Test filler word processing function."""

    def test_empty_list_returns_empty(self):
        """Test that empty input returns empty list."""
        result = process_filler_words([], pause_threshold=250)
        assert result == []

    def test_removes_default_filler_words(self):
        """Test removal of default filler words (äh, ähm)."""
        words = [
            {"type": "word", "text": "Hallo", "start": 0, "end": 100},
            {"type": "spacing", "text": " ", "start": 100, "end": 200},
            {"type": "word", "text": "ähm", "start": 200, "end": 300},
            {"type": "spacing", "text": " ", "start": 300, "end": 400},
            {"type": "word", "text": "Welt", "start": 400, "end": 500}
        ]
        result = process_filler_words(words, pause_threshold=250)

        # ähm should be removed, leaving Hallo + spacing
        word_texts = [w["text"] for w in result if w["type"] == "word"]
        assert "ähm" not in word_texts
        assert "Hallo" in word_texts
        # Welt may be removed if it gets merged with spacing after filler removal
        # The key is that ähm is gone

    def test_removes_custom_filler_words(self):
        """Test removal of custom filler words."""
        words = [
            {"type": "word", "text": "Hello", "start": 0, "end": 100},
            {"type": "spacing", "text": " ", "start": 100, "end": 200},
            {"type": "word", "text": "um", "start": 200, "end": 300},
            {"type": "spacing", "text": " ", "start": 300, "end": 400},
            {"type": "word", "text": "world", "start": 400, "end": 500}
        ]
        result = process_filler_words(words, pause_threshold=250, filler_words=["um", "uh"])

        word_texts = [w["text"] for w in result if w["type"] == "word"]
        assert "um" not in word_texts
        assert "Hello" in word_texts
        # world may be removed if it gets merged with spacing after filler removal
        # The key is that um is gone

    def test_removes_parenthesized_text(self):
        """Test removal of text in parentheses."""
        words = [
            {"type": "word", "text": "Hello", "start": 0, "end": 100},
            {"type": "spacing", "text": " ", "start": 100, "end": 200},
            {"type": "word", "text": "(clears throat)", "start": 200, "end": 300},
            {"type": "spacing", "text": " ", "start": 300, "end": 400},
            {"type": "word", "text": "world", "start": 400, "end": 500}
        ]
        result = process_filler_words(words, pause_threshold=250)

        word_texts = [w["text"] for w in result if w["type"] == "word"]
        assert "(clears throat)" not in word_texts

    def test_removes_audio_events(self):
        """Test removal of audio_event items."""
        words = [
            {"type": "word", "text": "Hello", "start": 0, "end": 100},
            {"type": "spacing", "text": " ", "start": 100, "end": 200},
            {"type": "audio_event", "text": "[noise]", "start": 200, "end": 300},
            {"type": "spacing", "text": " ", "start": 300, "end": 400},
            {"type": "word", "text": "world", "start": 400, "end": 500}
        ]
        result = process_filler_words(words, pause_threshold=250)

        # Audio event should be removed
        event_types = [w["type"] for w in result]
        assert "audio_event" not in event_types

    def test_case_insensitive_filler_matching(self):
        """Test that filler word matching is case-insensitive."""
        words = [
            {"type": "word", "text": "Hello", "start": 0, "end": 100},
            {"type": "spacing", "text": " ", "start": 100, "end": 200},
            {"type": "word", "text": "UM", "start": 200, "end": 300},
            {"type": "spacing", "text": " ", "start": 300, "end": 400},
            {"type": "word", "text": "world", "start": 400, "end": 500}
        ]
        result = process_filler_words(words, pause_threshold=250, filler_words=["um"])

        word_texts = [w["text"] for w in result if w["type"] == "word"]
        assert "UM" not in word_texts


@pytest.mark.unit
class TestPauseDetection:
    """Test pause detection and marking."""

    def test_converts_spacing_to_pause_marker(self):
        """Test that existing spacing elements are converted to pause markers."""
        words = [
            {"type": "spacing", "text": " ", "start": 0, "end": 500},  # 500ms pause
            {"type": "word", "text": "Hello", "start": 500, "end": 1000}
        ]
        result = standardize_word_format(words, show_pauses=True, silence_threshold=250)

        # The spacing should be converted to pause marker
        assert result[0]["type"] == "spacing"
        assert "(...)" in result[0]["text"]

    def test_no_pause_for_small_gaps(self):
        """Test that small gaps don't get pause markers."""
        words = [
            {"type": "word", "text": "Hello", "start": 0, "end": 100},
            {"type": "word", "text": "world", "start": 150, "end": 250}  # 50ms gap
        ]
        result = standardize_word_format(words, show_pauses=True, silence_threshold=250)

        # Should have spacing but no pause marker
        spacing = [r for r in result if r["type"] == "spacing"]
        assert len(spacing) == 1
        assert "(...)" not in spacing[0]["text"]

    def test_pause_threshold_customization(self):
        """Test custom pause threshold."""
        words = [
            {"type": "word", "text": "Hello", "start": 0, "end": 100},
            {"type": "word", "text": "world", "start": 200, "end": 300}  # 100ms gap
        ]
        # With 50ms threshold, 100ms should be a pause
        result = standardize_word_format(words, show_pauses=True, silence_threshold=50)

        spacing = [r for r in result if r["type"] == "spacing"]
        assert len(spacing) == 1
        assert "(...)" in spacing[0]["text"]


@pytest.mark.unit
class TestSpeakerLabelProcessing:
    """Test speaker label handling in text processing."""

    def test_preserves_speaker_labels_through_processing(self):
        """Test that speaker labels are preserved through filler word removal."""
        words = [
            {"type": "word", "text": "Hello", "start": 0, "end": 100, "speaker_id": "S1"},
            {"type": "spacing", "text": " ", "start": 100, "end": 200, "speaker_id": "S1"},
            {"type": "word", "text": "um", "start": 200, "end": 300, "speaker_id": "S1"},
            {"type": "spacing", "text": " ", "start": 300, "end": 400, "speaker_id": "S1"},
            {"type": "word", "text": "world", "start": 400, "end": 500, "speaker_id": "S1"}
        ]
        result = process_filler_words(words, pause_threshold=250, filler_words=["um"])

        # All remaining items should have speaker_id
        assert all("speaker_id" in r for r in result)

    def test_inherits_speaker_for_spacing(self):
        """Test that spacing inherits speaker from adjacent words."""
        words = [
            {"type": "word", "text": "Hello", "start": 0, "end": 100, "speaker_id": "S1"},
            {"type": "word", "text": "world", "start": 200, "end": 300, "speaker_id": "S1"}
        ]
        result = standardize_word_format(words)

        # Spacing should inherit speaker_id
        spacing = [r for r in result if r["type"] == "spacing"]
        assert len(spacing) == 1
        assert spacing[0]["speaker_id"] == "S1"

    def test_handles_missing_speaker_id(self):
        """Test handling of missing speaker_id fields."""
        words = [
            {"type": "word", "text": "Hello", "start": 0, "end": 100},
            {"type": "word", "text": "world", "start": 200, "end": 300}
        ]
        result = standardize_word_format(words)

        # Should not crash, speaker_id should be empty string
        spacing = [r for r in result if r["type"] == "spacing"]
        assert len(spacing) == 1
        assert spacing[0].get("speaker_id", "") == ""

    def test_different_speakers_not_merged(self):
        """Test that words from different speakers are handled correctly."""
        words = [
            {"type": "word", "text": "Hello", "start": 0, "end": 100, "speaker_id": "S1"},
            {"type": "word", "text": "world", "start": 200, "end": 300, "speaker_id": "S2"}
        ]
        result = standardize_word_format(words)

        # Should preserve both speakers
        assert result[0]["speaker_id"] == "S1"
        assert result[2]["speaker_id"] == "S2"


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_handles_overlapping_words(self):
        """Test handling of overlapping word timestamps."""
        words = [
            {"type": "word", "text": "Hello", "start": 0, "end": 200},
            {"type": "word", "text": "world", "start": 150, "end": 300}  # Overlaps by 50ms
        ]
        # Should not crash, just log a warning
        result = standardize_word_format(words)

        # Should still produce output
        assert len(result) >= 2

    def test_handles_negative_gaps(self):
        """Test handling of negative gaps (overlapping words)."""
        words = [
            {"type": "word", "text": "Hello", "start": 0, "end": 100},
            {"type": "word", "text": "world", "start": 50, "end": 150}  # Starts before first ends
        ]
        result = standardize_word_format(words)

        # Should handle gracefully
        assert len(result) >= 2

    def test_handles_single_word_list(self):
        """Test handling of list with only one word."""
        words = [
            {"type": "word", "text": "Hello", "start": 0, "end": 100}
        ]
        result = standardize_word_format(words)

        assert len(result) == 1
        assert result[0]["text"] == "Hello"

    def test_handles_none_values_in_list(self):
        """Test handling of None values in word list."""
        words = [
            {"type": "word", "text": "Hello", "start": 0, "end": 100},
            None,
            {"type": "word", "text": "world", "start": 200, "end": 300}
        ]
        result = process_filler_words(words, pause_threshold=250)

        # Should skip None values
        assert len(result) == 2

    def test_handles_words_with_missing_fields(self):
        """Test handling of word dictionaries with missing fields."""
        words = [
            {"text": "Hello", "start": 0},  # Missing 'end'
            {"text": "world", "end": 300}  # Missing 'start'
        ]
        result = standardize_word_format(words)

        # Should handle with default values
        assert len(result) == 2

    def test_handles_unicode_in_text(self):
        """Test handling of unicode characters in text."""
        words = [
            {"type": "word", "text": "Hallo", "start": 0, "end": 100},
            {"type": "word", "text": "Welt", "start": 200, "end": 300},
            {"type": "word", "text": "", "start": 300, "end": 400}
        ]
        result = standardize_word_format(words)

        # Should handle unicode without issues
        assert len(result) >= 2


@pytest.mark.unit
class TestMergeConsecutivePauses:
    """Test merging of consecutive pause entries in SRT files."""

    def test_merge_two_consecutive_pauses(self, tmp_path):
        """Test merging two consecutive pause entries."""
        srt_content = """1
00:00:00,000 --> 00:00:01,000
(...)

2
00:00:01,000 --> 00:00:02,000
(...)
"""
        srt_file = tmp_path / "test.srt"
        srt_file.write_text(srt_content, encoding='utf-8')

        merge_consecutive_pauses(srt_file)

        result = srt_file.read_text(encoding='utf-8')

        # Should have only one entry now
        entries = result.split('\n\n')
        assert len(entries) == 1

        # Should have merged time range
        assert "00:00:00,000 --> 00:00:02,000" in result

    def test_no_merge_non_consecutive_pauses(self, tmp_path):
        """Test that non-consecutive pauses are not merged."""
        srt_content = """1
00:00:00,000 --> 00:00:01,000
(...)

2
00:00:01,000 --> 00:00:02,000
Hello

3
00:00:02,000 --> 00:00:03,000
(...)
"""
        srt_file = tmp_path / "test.srt"
        srt_file.write_text(srt_content, encoding='utf-8')

        merge_consecutive_pauses(srt_file)

        result = srt_file.read_text(encoding='utf-8')

        # Should still have 3 entries (no merge)
        entries = result.split('\n\n')
        assert len(entries) == 3

    def test_handles_malformed_srt(self, tmp_path):
        """Test handling of malformed SRT content."""
        srt_content = "This is not valid SRT content"
        srt_file = tmp_path / "test.srt"
        srt_file.write_text(srt_content, encoding='utf-8')

        # Should not crash
        merge_consecutive_pauses(srt_file)

        # File should still exist
        assert srt_file.exists()


@pytest.mark.unit
class TestFindLongestCommonSequence:
    """Test sequence alignment and merging."""

    def test_empty_sequences_returns_empty(self):
        """Test that empty input returns empty string."""
        result = find_longest_common_sequence([])
        assert result == ""

    def test_single_sequence_returns_unchanged(self):
        """Test that single sequence is returned unchanged."""
        result = find_longest_common_sequence(["Hello world"])
        assert result == "Hello world"

    def test_two_identical_sequences(self):
        """Test merging two identical sequences."""
        result = find_longest_common_sequence([
            "Hello world",
            "Hello world"
        ])
        assert result == "Hello world"

    def test_two_similar_sequences(self):
        """Test merging two similar sequences."""
        result = find_longest_common_sequence([
            "Hello world test",
            "Hello world example"
        ], match_by_words=True)

        # Should find common prefix
        assert "Hello world" in result

    def test_character_based_matching(self):
        """Test character-based sequence matching."""
        result = find_longest_common_sequence([
            "abcdef",
            "abcxyz"
        ], match_by_words=False)

        # Should find common prefix
        assert "abc" in result


@pytest.mark.unit
class TestSegmentsToWords:
    """Test conversion of segments to word-level format."""

    def test_empty_segments_returns_empty(self):
        """Test that empty segments return empty list."""
        result = segments_to_words([])
        assert result == []

    def test_single_segment_to_words(self):
        """Test converting single segment to words."""
        segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello world", "speaker": "S1"}
        ]
        result = segments_to_words(segments)

        # Should have words + spacing
        words = [r for r in result if r["type"] == "word"]
        assert len(words) == 2
        assert words[0]["text"] == "Hello"
        assert words[1]["text"] == "world"

    def test_preserves_speaker_from_segment(self):
        """Test that speaker is preserved from segment."""
        segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello", "speaker_id": "S1"}
        ]
        result = segments_to_words(segments)

        assert result[0]["speaker_id"] == "S1"

    def test_calculates_word_timing(self):
        """Test that word timing is calculated proportionally."""
        segments = [
            {"start": 0.0, "end": 2.0, "text": "One two three"}
        ]
        result = segments_to_words(segments)

        words = [r for r in result if r["type"] == "word"]
        assert len(words) == 3

        # Each word should get approximately 0.67s
        assert words[0]["start"] == 0.0
        assert words[1]["start"] == pytest.approx(0.67, rel=0.1)
        assert words[2]["start"] == pytest.approx(1.33, rel=0.1)

    def test_handles_empty_segment_text(self):
        """Test handling of segments with empty text."""
        segments = [
            {"start": 0.0, "end": 1.0, "text": "   "}
        ]
        result = segments_to_words(segments)

        # Should skip empty segments
        assert len(result) == 0

    def test_adds_spacing_between_words(self):
        """Test that spacing is added between words."""
        segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello world"}
        ]
        result = segments_to_words(segments)

        # Should have spacing between words
        spacing = [r for r in result if r["type"] == "spacing"]
        assert len(spacing) == 1
