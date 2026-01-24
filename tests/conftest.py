"""
Pytest configuration and fixtures for audio_transcribe tests.

This module provides shared fixtures and utilities for testing,
including the LLM judge for non-deterministic output comparison.
"""
import json
import sys
from pathlib import Path
from typing import Generator, Optional, Tuple
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from audio_transcribe.utils.parsers import TranscriptionResult
from audio_transcribe.utils.config import ConfigManager


# Test fixtures directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"
AUDIO_FIXTURES_DIR = FIXTURES_DIR / "audio_files"
EXPECTED_OUTPUTS_DIR = FIXTURES_DIR / "expected_outputs"

# Use the existing test directory for audio files
TEST_AUDIO_DIR = Path(__file__).parent.parent / "test"

# Initialize config manager to load API keys
config_manager = ConfigManager()


@pytest.fixture(scope="session")
def api_keys():
    """
    Load API keys using ConfigManager.

    Returns dictionary with available API keys.
    """
    return {
        "openai": config_manager.get_api_key("openai"),
        "groq": config_manager.get_api_key("groq"),
        "assemblyai": config_manager.get_api_key("assemblyai"),
        "elevenlabs": config_manager.get_api_key("elevenlabs"),
        "gemini": config_manager.get_api_key("gemini"),
        "mistral": config_manager.get_api_key("mistral"),
    }


@pytest.fixture(scope="session")
def has_any_api_key(api_keys):
    """Check if at least one API key is available."""
    return any(key is not None for key in api_keys.values())


def llm_compare_texts(expected: str, actual: str, strict: bool = False) -> Tuple[bool, str]:
    """
    Compare transcription outputs using LLM as judge.

    API responses are non-deterministic (punctuation, capitalization vary).
    LLM evaluates semantic equivalence rather than string equality.

    Args:
        expected: Expected transcription text
        actual: Actual transcription text
        strict: If True, requires exact match (for word-for-word)

    Returns:
        Tuple of (passed: bool, reason: str)
    """
    # For now, use simple heuristic comparison
    # TODO: Implement actual LLM-based comparison using Groq/Mistral API

    if strict:
        # Word-for-word comparison (normalized)
        expected_words = set(expected.lower().split())
        actual_words = set(actual.lower().split())
        missing = expected_words - actual_words
        extra = actual_words - expected_words

        if missing or extra:
            return False, f"Strict mismatch - Missing: {missing}, Extra: {extra}"
        return True, "Exact match (normalized)"

    # Semantic comparison (relaxed)
    # Normalize: lowercase, remove punctuation, normalize whitespace
    import re

    def normalize(text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    expected_norm = normalize(expected)
    actual_norm = normalize(actual)

    # Calculate word overlap ratio
    expected_words = set(expected_norm.split())
    actual_words = set(actual_norm.split())

    if not expected_words:
        return True, "Empty expected text"

    overlap = expected_words & actual_words
    overlap_ratio = len(overlap) / len(expected_words)

    # Threshold: 80% word overlap for passing
    if overlap_ratio >= 0.8:
        return True, f"Semantic match (overlap: {overlap_ratio:.1%})"

    missing = expected_words - actual_words
    return False, f"Semantic mismatch - Missing words: {missing}, Overlap: {overlap_ratio:.1%}"


def llm_compare_transcriptions(
    expected_file: Path,
    actual: TranscriptionResult,
    api_name: str,
    sample_name: str,
    timing_tolerance: float = 0.2
) -> Tuple[bool, str]:
    """
    Compare actual transcription with expected output using LLM judge.

    Args:
        expected_file: Path to expected output file
        actual: Actual TranscriptionResult
        api_name: Name of API being tested
        sample_name: Name of test sample
        timing_tolerance: Timing tolerance in seconds (default: 0.2 = 200ms)

    Returns:
        Tuple of (passed: bool, reason: str)
    """
    # Read expected output
    if expected_file.suffix == ".json":
        with open(expected_file, encoding='utf-8') as f:
            expected_data = json.load(f)
        expected_text = expected_data.get("text", "")
        expected_words = expected_data.get("words", [])
    else:
        with open(expected_file, encoding='utf-8') as f:
            expected_text = f.read()
        expected_words = []

    # Compare text
    text_passed, text_reason = llm_compare_texts(expected_text, actual.text, strict=False)

    # If we have word timestamps in expected output, compare them
    if expected_words and actual.words:
        timing_passed, timing_reason = compare_word_timings(
            expected_words, actual.words, tolerance=timing_tolerance
        )
        return timing_passed, f"{text_reason} | {timing_reason}"

    return text_passed, text_reason


def compare_word_timings(
    expected_words: list,
    actual_words: list,
    tolerance: float = 0.2
) -> Tuple[bool, str]:
    """
    Compare word timings with tolerance for segment-level approximation.

    Args:
        expected_words: Expected word list with text, start, end
        actual_words: Actual word list with text, start, end
        tolerance: Timing tolerance in seconds (default: 0.2 = 200ms)

    Returns:
        Tuple of (passed: bool, reason: str)
    """
    # Filter out spacing entries (common in ElevenLabs output)
    expected_words_filtered = [w for w in expected_words if w.get("type") != "spacing"]

    if not expected_words_filtered:
        return True, "No word timestamps in expected output"

    if not actual_words:
        return False, "No word timestamps in actual output"

    # Normalize text for comparison
    def normalize_word(w):
        return w.get("text", "").lower().strip()

    expected_normalized = [normalize_word(w) for w in expected_words_filtered if normalize_word(w)]
    actual_normalized = [normalize_word(w) for w in actual_words if normalize_word(w)]

    # Check word count is roughly similar (allow 20% difference for filler words)
    expected_count = len(expected_normalized)
    actual_count = len(actual_normalized)
    count_ratio = min(expected_count, actual_count) / max(expected_count, actual_count)

    if count_ratio < 0.8:
        return False, f"Word count mismatch: expected {expected_count}, got {actual_count}"

    # Compare timing for words that match
    matches = 0
    timing_errors = []

    # Simple matching: compare words at similar positions
    min_len = min(len(expected_words_filtered), len(actual_words))

    for i in range(min_len):
        exp_word = expected_words_filtered[i]
        act_word = actual_words[i]

        if normalize_word(exp_word) != normalize_word(act_word):
            # Words don't match, skip timing check
            continue

        exp_start = exp_word.get("start", 0)
        exp_end = exp_word.get("end", 0)
        act_start = act_word.get("start", 0)
        act_end = act_word.get("end", 0)

        # Check if timing is within tolerance
        start_diff = abs(exp_start - act_start)
        end_diff = abs(exp_end - act_end)

        if start_diff <= tolerance and end_diff <= tolerance:
            matches += 1
        else:
            timing_errors.append(
                f"'{exp_word.get('text')}' start:{start_diff:.3f}s end:{end_diff:.3f}s"
            )

    match_ratio = matches / min_len if min_len > 0 else 0

    # Require 80% of words to match timing
    if match_ratio >= 0.8:
        return True, f"Timing match: {matches}/{min_len} words within {tolerance}s tolerance"

    return False, f"Timing mismatch: {matches}/{min_len} words matched. Errors: {timing_errors[:5]}"


@pytest.fixture
def sample_audio_files() -> dict:
    """
    Provide paths to sample audio files for testing.

    Returns dictionary mapping sample names to file paths.
    """
    files = {
        "speech": AUDIO_FIXTURES_DIR / "sample_speech.m4a",
        "multi_speaker": AUDIO_FIXTURES_DIR / "sample_multi_speaker.wav",
        "long_audio": AUDIO_FIXTURES_DIR / "sample_long_audio.wav",
        "video": AUDIO_FIXTURES_DIR / "sample_video.mkv",
        "multiple_langs": AUDIO_FIXTURES_DIR / "sample_multiple_langs.m4a",
    }

    # Filter to only existing files
    return {k: v for k, v in files.items() if v.exists()}


@pytest.fixture
def sample_audio_file(sample_audio_files) -> Optional[Path]:
    """Get a single sample audio file (speech)."""
    return sample_audio_files.get("speech")


@pytest.fixture
def expected_outputs() -> dict:
    """
    Provide paths to expected output files for testing.

    Returns dictionary mapping sample names to expected output paths.
    """
    return {
        "speech": {
            "txt": TEST_AUDIO_DIR / "expected-speech.txt",
            "json": TEST_AUDIO_DIR / "expected-speech.json",
        },
        "multi_speaker": {
            "txt": TEST_AUDIO_DIR / "expected-multi-speaker.txt",
            "json": TEST_AUDIO_DIR / "expected-multi-speaker.json",
        },
    }


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (require API keys)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (may call real APIs)"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests (fast, no external calls)"
    )


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """
    Provide a temporary directory for test outputs.

    Automatically cleaned up after test.
    """
    yield tmp_path


@pytest.fixture
def mock_transcription_result() -> TranscriptionResult:
    """
    Provide a mock TranscriptionResult for testing.

    Contains realistic sample data.
    """
    return TranscriptionResult(
        text="This is a sample transcription for testing purposes.",
        confidence=0.95,
        language="en",
        words=[
            {"text": "This", "start": 0.0, "end": 0.3},
            {"text": "is", "start": 0.3, "end": 0.5},
            {"text": "a", "start": 0.5, "end": 0.6},
            {"text": "sample", "start": 0.6, "end": 1.0},
            {"text": "transcription", "start": 1.0, "end": 1.6},
            {"text": "for", "start": 1.6, "end": 1.8},
            {"text": "testing", "start": 1.8, "end": 2.2},
            {"text": "purposes.", "start": 2.2, "end": 2.8},
        ],
        speakers=[],
        segments=[],
        api_name="mock"
    )
