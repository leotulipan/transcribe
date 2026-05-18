"""
Generate golden output files using ElevenLabs API.

This module creates expected output files for testing by transcribing
sample audio files with ElevenLabs (the most accurate API).

Run with: uv run pytest tests/acceptance/test_generate_golden_outputs.py::TestGenerateGoldenOutputs -v -s
"""
import json
import pytest
from pathlib import Path

from audio_transcribe.utils.api import get_api_instance

# Test fixtures directory
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
EXPECTED_OUTPUTS_DIR = FIXTURES_DIR / "expected_outputs"

# Ensure expected outputs directory exists
EXPECTED_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


@pytest.mark.integration
@pytest.mark.slow
class TestGenerateGoldenOutputs:
    """
    Generate golden output files using ElevenLabs API.

    These tests create the expected output files that other tests will compare against.
    Run this once to generate the golden outputs, then use those for regression testing.
    """

    def test_generate_speech_golden_outputs(self, api_keys, sample_audio_files):
        """Generate golden outputs for sample_speech.m4a using ElevenLabs."""
        elevenlabs_key = api_keys.get("elevenlabs")
        if not elevenlabs_key:
            pytest.skip("No ElevenLabs API key available")

        speech_file = sample_audio_files.get("speech")
        if not speech_file:
            pytest.skip("No sample_speech.m4a file found")

        api = get_api_instance("elevenlabs", elevenlabs_key)

        # Transcribe the audio file
        result = api.transcribe(speech_file)

        # Save text output
        txt_output = EXPECTED_OUTPUTS_DIR / "sample_speech.txt"
        with open(txt_output, 'w', encoding='utf-8') as f:
            f.write(result.text)
        print(f"\n[OK] Created golden text output: {txt_output}")

        # Save JSON output
        json_output = EXPECTED_OUTPUTS_DIR / "sample_speech.json"
        result.save(json_output)
        print(f"[OK] Created golden JSON output: {json_output}")

        # Verify the outputs were created
        assert txt_output.exists()
        assert json_output.exists()
        assert len(result.text) > 0
        assert len(result.words) > 0

    def test_generate_multi_speaker_golden_outputs(self, api_keys, sample_audio_files):
        """Generate golden outputs for sample_multi_speaker.wav using ElevenLabs."""
        elevenlabs_key = api_keys.get("elevenlabs")
        if not elevenlabs_key:
            pytest.skip("No ElevenLabs API key available")

        multi_speaker_file = sample_audio_files.get("multi_speaker")
        if not multi_speaker_file:
            pytest.skip("No sample_multi_speaker.wav file found")

        api = get_api_instance("elevenlabs", elevenlabs_key)

        # Transcribe with diarization enabled
        result = api.transcribe(multi_speaker_file, diarize=True)

        # Save text output
        txt_output = EXPECTED_OUTPUTS_DIR / "sample_multi_speaker.txt"
        with open(txt_output, 'w', encoding='utf-8') as f:
            f.write(result.text)
        print(f"\n[OK] Created golden text output: {txt_output}")

        # Save JSON output
        json_output = EXPECTED_OUTPUTS_DIR / "sample_multi_speaker.json"
        result.save(json_output)
        print(f"[OK] Created golden JSON output: {json_output}")

        # Verify the outputs were created
        assert txt_output.exists()
        assert json_output.exists()
        assert len(result.text) > 0
        # Multi-speaker should have speaker info
        if result.speakers:
            assert len(result.speakers) > 0

    def test_generate_long_audio_golden_outputs(self, api_keys, sample_audio_files):
        """Generate golden outputs for sample_long_audio.wav using ElevenLabs."""
        elevenlabs_key = api_keys.get("elevenlabs")
        if not elevenlabs_key:
            pytest.skip("No ElevenLabs API key available")

        long_audio_file = sample_audio_files.get("long_audio")
        if not long_audio_file:
            pytest.skip("No sample_long_audio.wav file found")

        api = get_api_instance("elevenlabs", elevenlabs_key)

        # Transcribe the long audio file
        result = api.transcribe(long_audio_file)

        # Save text output
        txt_output = EXPECTED_OUTPUTS_DIR / "sample_long_audio.txt"
        with open(txt_output, 'w', encoding='utf-8') as f:
            f.write(result.text)
        print(f"\n[OK] Created golden text output: {txt_output}")

        # Save JSON output
        json_output = EXPECTED_OUTPUTS_DIR / "sample_long_audio.json"
        result.save(json_output)
        print(f"[OK] Created golden JSON output: {json_output}")

        # Verify the outputs were created
        assert txt_output.exists()
        assert json_output.exists()
        assert len(result.text) > 0

    def test_generate_video_golden_outputs(self, api_keys, sample_audio_files):
        """Generate golden outputs for sample_video.mkv using ElevenLabs."""
        elevenlabs_key = api_keys.get("elevenlabs")
        if not elevenlabs_key:
            pytest.skip("No ElevenLabs API key available")

        video_file = sample_audio_files.get("video")
        if not video_file:
            pytest.skip("No sample_video.mkv file found")

        api = get_api_instance("elevenlabs", elevenlabs_key)

        # Transcribe the video file
        result = api.transcribe(video_file)

        # Save text output
        txt_output = EXPECTED_OUTPUTS_DIR / "sample_video.txt"
        with open(txt_output, 'w', encoding='utf-8') as f:
            f.write(result.text)
        print(f"\n[OK] Created golden text output: {txt_output}")

        # Save JSON output
        json_output = EXPECTED_OUTPUTS_DIR / "sample_video.json"
        result.save(json_output)
        print(f"[OK] Created golden JSON output: {json_output}")

        # Verify the outputs were created
        assert txt_output.exists()
        assert json_output.exists()
        assert len(result.text) > 0

    def test_generate_multiple_langs_golden_outputs(self, api_keys, sample_audio_files):
        """Generate golden outputs for sample_multiple_langs.m4a using ElevenLabs."""
        elevenlabs_key = api_keys.get("elevenlabs")
        if not elevenlabs_key:
            pytest.skip("No ElevenLabs API key available")

        multiple_langs_file = sample_audio_files.get("multiple_langs")
        if not multiple_langs_file:
            pytest.skip("No sample_multiple_langs.m4a file found")

        api = get_api_instance("elevenlabs", elevenlabs_key)

        # Transcribe the multi-language audio file (auto-detect language)
        result = api.transcribe(multiple_langs_file)

        # Save text output
        txt_output = EXPECTED_OUTPUTS_DIR / "sample_multiple_langs.txt"
        with open(txt_output, 'w', encoding='utf-8') as f:
            f.write(result.text)
        print(f"\n[OK] Created golden text output: {txt_output}")

        # Save JSON output
        json_output = EXPECTED_OUTPUTS_DIR / "sample_multiple_langs.json"
        result.save(json_output)
        print(f"[OK] Created golden JSON output: {json_output}")

        # Verify the outputs were created
        assert txt_output.exists()
        assert json_output.exists()
        assert len(result.text) > 0
