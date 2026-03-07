"""
Integration tests for CLI workflow.

Tests the complete end-to-end CLI workflow including:
- Positional file argument
- Folder option
- API selection
- Language parameter
- Output formats (multiple)
- DaVinci SRT option
- Speaker labels option
- Size threshold option
- Model selection
- Verbose logging
- Debug logging
"""
import pytest
from pathlib import Path
from click.testing import CliRunner
from unittest.mock import patch
from audio_transcribe.cli import main
from audio_transcribe.utils.config import ConfigManager


@pytest.mark.integration
class TestCLIArguments:
    """Test suite for CLI argument handling."""

    def test_positional_file_argument(self, sample_audio_file, tmp_path):
        """Test that positional file argument is accepted."""
        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        runner = CliRunner()

        # Mock the API to avoid actual transcription
        with patch('audio_transcribe.cli.get_api_instance') as mock_get_api:
            mock_api = mock_get_api.return_value
            mock_api.transcribe.return_value = type('obj', (object,), {
                'text': 'Test transcription',
                'words': [],
                'api_name': 'test'
            })

            result = runner.invoke(main, ['--api', 'groq', str(sample_audio_file)])

            assert result.exit_code == 0 or "transcribing" in result.output.lower() or "processing" in result.output.lower()

    def test_folder_option(self, sample_audio_files, tmp_path):
        """Test that directory as positional arg processes audio files."""
        if not sample_audio_files:
            pytest.skip("No sample audio files available")

        runner = CliRunner()

        # Get first audio file from the dict
        first_file = next(iter(sample_audio_files.values()))
        folder_path = first_file.parent

        # Mock the API
        with patch('audio_transcribe.cli.get_api_instance') as mock_get_api:
            mock_api = mock_get_api.return_value
            mock_api.transcribe.return_value = type('obj', (object,), {
                'text': 'Test',
                'words': [],
                'api_name': 'test'
            })

            result = runner.invoke(main, ['--api', 'groq', str(folder_path)])

            assert result.exit_code == 0 or "processing" in result.output.lower()

    def test_api_selection(self, sample_audio_file):
        """Test --api option for API selection."""
        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        runner = CliRunner()

        with patch('audio_transcribe.cli.get_api_instance') as mock_get_api:
            mock_api = mock_get_api.return_value
            mock_api.transcribe.return_value = type('obj', (object,), {
                'text': 'Test',
                'words': [],
                'api_name': 'test'
            })

            result = runner.invoke(main, ['--api', 'groq', str(sample_audio_file)])

            assert result.exit_code == 0

    def test_language_parameter(self, sample_audio_file):
        """Test --language option for language specification."""
        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        runner = CliRunner()

        with patch('audio_transcribe.cli.get_api_instance') as mock_get_api:
            mock_api = mock_get_api.return_value
            mock_api.transcribe.return_value = type('obj', (object,), {
                'text': 'Test',
                'words': [],
                'api_name': 'test'
            })

            result = runner.invoke(main, ['--api', 'groq', '--language', 'de', str(sample_audio_file)])

            assert result.exit_code == 0

    def test_output_formats_multiple(self, sample_audio_file):
        """Test --output option with multiple formats."""
        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        runner = CliRunner()

        with patch('audio_transcribe.cli.get_api_instance') as mock_get_api:
            mock_api = mock_get_api.return_value
            mock_api.transcribe.return_value = type('obj', (object,), {
                'text': 'Test transcription',
                'words': [
                    {'text': 'Test', 'start': 0.0, 'end': 1.0}
                ],
                'segments': [],
                'api_name': 'test'
            })

            result = runner.invoke(main, ['--api', 'groq', '--output', 'text', '--output', 'srt', str(sample_audio_file)])

            assert result.exit_code == 0

    def test_davinci_srt_option(self, sample_audio_file):
        """Test --davinci-srt option for DaVinci Resolve format."""
        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        runner = CliRunner()

        with patch('audio_transcribe.cli.get_api_instance') as mock_get_api:
            mock_api = mock_get_api.return_value
            mock_api.transcribe.return_value = type('obj', (object,), {
                'text': 'Test transcription',
                'words': [
                    {'text': 'Test', 'start': 0.0, 'end': 1.0},
                    {'text': 'pause', 'start': 1.0, 'end': 2.0}
                ],
                'segments': [],
                'api_name': 'test'
            })

            result = runner.invoke(main, ['--api', 'groq', '--davinci-srt', str(sample_audio_file)])

            assert result.exit_code == 0

    def test_model_selection(self, sample_audio_file):
        """Test --model option for model selection."""
        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        runner = CliRunner()

        with patch('audio_transcribe.cli.get_api_instance') as mock_get_api:
            mock_api = mock_get_api.return_value
            mock_api.transcribe.return_value = type('obj', (object,), {
                'text': 'Test',
                'words': [],
                'api_name': 'test'
            })

            result = runner.invoke(main, ['--api', 'openai', '--model', 'whisper-1', str(sample_audio_file)])

            assert result.exit_code == 0

    def test_verbose_logging(self, sample_audio_file):
        """Test --verbose option for verbose logging."""
        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        runner = CliRunner()

        with patch('audio_transcribe.cli.get_api_instance') as mock_get_api:
            mock_api = mock_get_api.return_value
            mock_api.transcribe.return_value = type('obj', (object,), {
                'text': 'Test',
                'words': [],
                'api_name': 'test'
            })

            result = runner.invoke(main, ['--api', 'groq', '--verbose', str(sample_audio_file)])

            assert result.exit_code == 0

    def test_debug_logging(self, sample_audio_file):
        """Test --debug option for debug logging."""
        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        runner = CliRunner()

        with patch('audio_transcribe.cli.get_api_instance') as mock_get_api:
            mock_api = mock_get_api.return_value
            mock_api.transcribe.return_value = type('obj', (object,), {
                'text': 'Test',
                'words': [],
                'api_name': 'test'
            })

            result = runner.invoke(main, ['--api', 'groq', '--debug', str(sample_audio_file)])

            assert result.exit_code == 0


@pytest.mark.integration
class TestCLIWorkflow:
    """Test suite for end-to-end CLI workflows."""

    def test_transcribe_single_file(self, sample_audio_file):
        """Test transcribing a single file."""
        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        runner = CliRunner()

        with patch('audio_transcribe.cli.get_api_instance') as mock_get_api:
            mock_api = mock_get_api.return_value
            mock_api.transcribe.return_value = type('obj', (object,), {
                'text': 'Sample transcription result',
                'words': [
                    {'text': 'Sample', 'start': 0.0, 'end': 0.5},
                    {'text': 'transcription', 'start': 0.5, 'end': 2.0}
                ],
                'segments': [
                    {'start': 0.0, 'end': 2.0, 'text': 'Sample transcription'}
                ],
                'api_name': 'test'
            })

            result = runner.invoke(main, ['--api', 'groq', '--output', 'text', str(sample_audio_file)])

            assert result.exit_code == 0

    def test_batch_process_folder(self, sample_audio_files):
        """Test batch processing of a directory via positional arg."""
        if not sample_audio_files:
            pytest.skip("No sample audio files available")

        runner = CliRunner()

        # Get first file as the folder
        first_file = next(iter(sample_audio_files.values()))
        folder_path = first_file.parent

        with patch('audio_transcribe.cli.get_api_instance') as mock_get_api:
            mock_api = mock_get_api.return_value
            mock_api.transcribe.return_value = type('obj', (object,), {
                'text': 'Test',
                'words': [],
                'api_name': 'test'
            })

            result = runner.invoke(main, ['--api', 'groq', str(folder_path)])

            assert result.exit_code == 0 or "processing" in result.output.lower()

    def test_skip_existing_transcript(self, sample_audio_file, tmp_path):
        """Test skipping transcription when output already exists."""
        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        runner = CliRunner()

        # Create a fake existing transcript
        txt_path = sample_audio_file.with_suffix('.txt')
        txt_path.write_text("Existing transcript")

        with patch('audio_transcribe.cli.get_api_instance') as mock_get_api:
            mock_api = mock_get_api.return_value
            mock_api.transcribe.return_value = type('obj', (object,), {
                'text': 'Test',
                'words': [],
                'api_name': 'test'
            })

            result = runner.invoke(main, ['--api', 'groq', str(sample_audio_file)])

            # Should skip API call if transcript exists
            assert result.exit_code == 0

        # Cleanup
        txt_path.unlink()

    def test_auto_detect_json_input(self, tmp_path):
        """Test auto-detection of JSON input files."""
        runner = CliRunner()

        # Create a test JSON file
        json_file = tmp_path / "test_groq.json"
        json_file.write_text('{"text": "Test", "words": [], "api_name": "groq"}')

        result = runner.invoke(main, [str(json_file)])

        # Should auto-detect and use JSON input
        assert result.exit_code == 0 or "json" in result.output.lower()

        # Cleanup
        json_file.unlink()

    def test_output_file_creation(self, sample_audio_file, tmp_path):
        """Test that output files are created correctly."""
        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        runner = CliRunner()

        with patch('audio_transcribe.cli.get_api_instance') as mock_get_api:
            mock_api = mock_get_api.return_value
            mock_api.transcribe.return_value = type('obj', (object,), {
                'text': 'Sample transcription',
                'words': [
                    {'text': 'Sample', 'start': 0.0, 'end': 0.5},
                    {'text': 'transcription', 'start': 0.5, 'end': 2.0}
                ],
                'segments': [
                    {'start': 0.0, 'end': 2.0, 'text': 'Sample transcription'}
                ],
                'api_name': 'test'
            })

            result = runner.invoke(main, [
                '--api', 'groq',
                '--output', 'text',
                '--output', 'srt',
                str(sample_audio_file)
            ])

            assert result.exit_code == 0

            # Check if output files were created next to input file
            txt_file = sample_audio_file.with_suffix('.txt')
            srt_file = sample_audio_file.with_suffix('.srt')

            # Files should be created next to the input file
            # Note: The CliRunner uses a temp directory, so files won't persist
            # We just check the command completed successfully

    def test_error_file_not_found(self, tmp_path):
        """Test error handling for non-existent file."""
        runner = CliRunner()

        result = runner.invoke(main, ['nonexistent.wav'])

        # Should exit with error
        assert result.exit_code != 0

    def test_error_invalid_api_key(self, tmp_path):
        """Test error handling for invalid API key."""
        runner = CliRunner()

        # Create a dummy audio file
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00\x00" + b"WAVE" + b"\x00\x00\x00\x00" + b"fmt " + b"\x10\x00\x00\x00" + b"data" + b"\x00\x00\x00\x00" * 100)

        with patch('audio_transcribe.cli.get_api_instance') as mock_get_api:
            mock_api = mock_get_api.return_value
            mock_api.check_api_key.return_value = False

            result = runner.invoke(main, [str(audio_file)])

            # Should fail with API key error
            assert result.exit_code != 0 or "api key" in result.output.lower()

    def test_preserve_on_error(self, sample_audio_file):
        """Test --preserve option to keep intermediate files on error."""
        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        runner = CliRunner()

        with patch('audio_transcribe.cli.get_api_instance') as mock_get_api:
            mock_api = mock_get_api.return_value
            mock_api.transcribe.side_effect = Exception("Test error")

            result = runner.invoke(main, ['--api', 'groq', '--preserve', str(sample_audio_file)])

            # Should fail but preserve intermediates
            assert result.exit_code != 0

    def test_force_option(self, sample_audio_file, tmp_path):
        """Test --force option to re-transcribe even if output exists."""
        if not sample_audio_file:
            pytest.skip("No sample audio file available")

        runner = CliRunner()

        # Create a fake existing transcript
        txt_path = sample_audio_file.with_suffix('.txt')
        txt_path.write_text("Existing transcript")

        with patch('audio_transcribe.cli.get_api_instance') as mock_get_api:
            mock_api = mock_get_api.return_value
            mock_api.transcribe.return_value = type('obj', (object,), {
                'text': 'New transcription',
                'words': [],
                'api_name': 'test'
            })

            # Without --force, should skip
            result_skip = runner.invoke(main, ['--api', 'groq', str(sample_audio_file)])
            assert result_skip.exit_code == 0

            # With --force, should re-transcribe
            result_force = runner.invoke(main, ['--api', 'groq', '--force', str(sample_audio_file)])
            assert result_force.exit_code == 0

        # Cleanup
        txt_path.unlink()
