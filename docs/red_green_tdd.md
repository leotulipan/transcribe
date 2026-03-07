# Red/Green TDD for Audio Transcribe

## Setup

```bash
# Install all dependencies including dev/test extras
uv sync --all-extras

# Or install dev dependencies explicitly
uv pip install pytest pytest-cov pytest-mock pytest-xdist
```

### Running tests

```bash
# Full run (unit only — fast, no API keys needed)
uv run python -m pytest tests/unit/ -v

# Stop on first failure
uv run python -m pytest tests/ -v -x

# Run one test by name
uv run python -m pytest tests/ -v -k "test_create_standard_srt"

# Run one test class
uv run python -m pytest tests/ -v -k "TestCreateSRTFile"

# Run by marker
uv run python -m pytest tests/ -v -m unit          # fast, no externals
uv run python -m pytest tests/ -v -m integration   # needs API keys
uv run python -m pytest tests/ -v -m "not slow"    # skip slow tests

# Parallel execution (pytest-xdist)
uv run python -m pytest tests/unit/ -v -n auto

# With coverage
uv run python -m pytest tests/ --cov=audio_transcribe --cov-report=term-missing
```


## Project structure

```
Transcribe/
  audio_transcribe/           # Source code
    cli.py
    utils/
      api/                    # Provider implementations
      parsers.py              # TranscriptionResult
      formatters.py           # Output generators
      config.py               # ConfigManager
      models.py               # MODEL_REGISTRY
      adapters.py
    transcribe_helpers/
      audio_processing.py
      output_formatters.py
      text_processing.py
      language_utils.py
      chunking.py
      pyav_backend.py
    tui/
  tests/
    conftest.py               # Shared fixtures, markers, LLM judge
    unit/                     # Fast tests, no external calls
      test_parsers.py
      test_formatters.py
      test_text_processing.py
      test_language_utils.py
      test_config.py
      test_audio_processing.py
      test_chunking.py
      test_intermediate_files.py
      test_pyav_backend.py
    integration/              # Needs API keys or ffmpeg
      test_cli.py
      test_api_integration_groq.py
      test_api_integration_openai.py
      test_api_integration_assemblyai.py
      test_api_integration_elevenlabs.py
      test_api_integration_gemini.py
      test_api_integration_mistral.py
      test_model_capabilities.py
      test_output_format_fallback.py
    acceptance/               # End-to-end acceptance scenarios (future)
    fixtures/
      audio_files/            # Sample audio/video for tests
      expected_outputs/       # Golden files for comparison
    scripts/
      generate_test_audio.py  # Auto-generates silent/stereo test fixtures
  test/                       # Legacy manual test files + expected outputs
```

### File naming conventions

- Test files: `test_<module>.py` — mirrors the source module name
- Test classes: `Test<Feature>` — group related tests
- Test functions: `test_<behavior>_<expected_outcome>` — reads like a sentence
- Fixtures: `conftest.py` in each test directory (pytest auto-discovers)


## The workflow

### 1. RED — Write the failing test FIRST

Before touching implementation, write the test for exactly what you need:

```python
def test_parse_gemini_extracts_text_and_words():
    raw_data = {
        "text": "Hello world",
        "results": [{"alternatives": [{"words": [
            {"word": "Hello", "startTime": "0s", "endTime": "0.5s"},
            {"word": "world", "startTime": "0.5s", "endTime": "1.0s"},
        ]}]}]
    }
    result = parse_gemini_format(raw_data)

    assert result.text == "Hello world"
    assert len(result.words) >= 2
    assert result.words[0]["text"] == "Hello"
```

Run it. It fails — `ImportError` because `parse_gemini_format` doesn't exist.
That's RED.

### 2. GREEN — Write the minimum code to pass

Don't gold-plate. Write just enough:

```python
def parse_gemini_format(raw_data: dict) -> TranscriptionResult:
    text = raw_data.get("text", "")
    words = []
    for result in raw_data.get("results", []):
        for alt in result.get("alternatives", []):
            for w in alt.get("words", []):
                words.append({
                    "text": w["word"],
                    "start": float(w["startTime"].rstrip("s")),
                    "end": float(w["endTime"].rstrip("s")),
                    "type": "word",
                })
    return TranscriptionResult(text=text, words=words, api_name="gemini")
```

Run again. Green. Move to the next test.

### 3. Repeat — one test at a time

Each new test drives a new piece of behavior. Only refactor when you have
green tests to protect you.


## Markers

Custom markers are registered in `conftest.py`:

| Marker              | Meaning                                      |
|----------------------|----------------------------------------------|
| `@pytest.mark.unit`           | Fast, no external calls               |
| `@pytest.mark.integration`    | Needs API keys, may call real APIs     |
| `@pytest.mark.slow`           | Long-running (real API calls)          |
| `@pytest.mark.requires_ffmpeg`| Needs ffmpeg installed                 |

Tests marked `requires_ffmpeg` are auto-skipped when ffmpeg is not on PATH
(handled by `pytest_collection_modifyitems` in `conftest.py`).

Integration tests should guard with `pytest.skip()` when keys are missing:

```python
def test_transcribe(self, sample_audio_file, api_keys):
    api_key = api_keys.get("groq")
    if not api_key:
        pytest.skip("No Groq API key available")
```


## Common test patterns

### Test a pure function directly

```python
def test_format_time_ms_converts_correctly():
    from audio_transcribe.transcribe_helpers.output_formatters import format_time_ms
    assert format_time_ms(0) == "00:00:00,000"
    assert format_time_ms(3661500) == "01:01:01,500"
```

### Test with parametrize for multiple cases

```python
@pytest.mark.parametrize("api_name,expected_limit", [
    ("groq", 25),
    ("openai", 25),
    ("assemblyai", 200),
    ("elevenlabs", 1000),
])
def test_api_file_size_limits(api_name, expected_limit):
    assert get_api_file_size_limit(api_name) == expected_limit
```

### Test file I/O with tmp_path

```python
def test_create_text_from_result(self, tmp_path):
    result = TranscriptionResult(text="Hello world", api_name="test")
    output_file = tmp_path / "test.txt"

    create_text_file(result, output_file)

    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")
    assert "Hello world" in content
```

### Test exceptions

```python
def test_raises_on_long_audio(self, long_audio):
    with pytest.raises(RuntimeError, match="exceeds maximum allowed length"):
        check_audio_length(long_audio, max_length=100)

def test_raises_on_invalid_file(self, tmp_path):
    invalid = tmp_path / "invalid.wav"
    invalid.write_text("not audio")
    with pytest.raises(Exception):
        optimize_audio_for_api(invalid, "groq")
```

### Test with monkeypatch / unittest.mock

```python
def test_cli_uses_correct_api(self, sample_audio_file):
    runner = CliRunner()
    with patch("audio_transcribe.cli.get_api_instance") as mock_get_api:
        mock_api = mock_get_api.return_value
        mock_api.transcribe.return_value = TranscriptionResult(
            text="Test", words=[], api_name="test"
        )
        result = runner.invoke(main, ["--api", "groq", str(sample_audio_file)])
        assert result.exit_code == 0
```

### Test CLI with Click's CliRunner

```python
from click.testing import CliRunner
from audio_transcribe.cli import main

def test_error_file_not_found():
    runner = CliRunner()
    result = runner.invoke(main, ["nonexistent.wav"])
    assert result.exit_code != 0
```

### Test non-deterministic API output (LLM judge)

Transcription APIs return non-deterministic text (punctuation, casing varies).
Use the `llm_compare_texts` helper from `conftest.py`:

```python
from tests.conftest import llm_compare_texts

def test_transcription_semantically_matches(actual_text):
    expected = "Hello world, how are you?"
    passed, reason = llm_compare_texts(expected, actual_text, strict=False)
    assert passed, reason
```

- `strict=False` — 80% word overlap after normalization (punctuation removed)
- `strict=True` — exact word set match (case-insensitive)

### Test audio processing (requires ffmpeg)

```python
@pytest.mark.requires_ffmpeg
def test_flac_conversion(self, sample_audio_file, tmp_path):
    result = convert_to_flac(sample_audio_file)
    assert Path(result).exists()
    assert Path(result).suffix == ".flac"
```

Use the `AudioSegment` factory for synthetic test audio:

```python
@pytest.fixture
def short_audio(self, tmp_path):
    audio = AudioSegment.silent(duration=5000)  # 5 seconds
    path = tmp_path / "short.wav"
    audio.export(str(path), format="wav")
    return path
```


## Fixtures reference

### Session-scoped (conftest.py)

| Fixture                     | Description                                    |
|-----------------------------|------------------------------------------------|
| `api_keys`                  | Dict of API keys loaded via ConfigManager      |
| `has_any_api_key`           | Bool — at least one key available              |
| `generate_test_fixtures`    | Auto-generates silent/stereo WAVs if missing   |
| `ffmpeg_available`          | Bool — ffmpeg on PATH                          |

### Function-scoped (conftest.py)

| Fixture                     | Description                                    |
|-----------------------------|------------------------------------------------|
| `sample_audio_files`        | Dict of sample name -> Path (existing only)    |
| `sample_audio_file`         | Single speech sample (m4a)                     |
| `expected_outputs`          | Dict of sample name -> {txt, json} paths       |
| `temp_output_dir`           | `tmp_path` alias, auto-cleaned                 |
| `mock_transcription_result` | Realistic `TranscriptionResult` with words     |

### Creating test-local fixtures

For test-specific data, define fixtures in the test file or a local `conftest.py`:

```python
@pytest.fixture
def srt_with_pauses(tmp_path):
    content = "1\n00:00:00,000 --> 00:00:01,000\n(...)\n\n"
    path = tmp_path / "test.srt"
    path.write_text(content, encoding="utf-8")
    return path
```


## Key rules

1. **Test behavior, not implementation.** Test what a function returns, not how
   it works internally. If you refactor the internals, tests should still pass.

2. **One assertion focus per test.** `test_parse_returns_correct_total` should
   only assert the total — not also check the filename or date.

3. **Name tests like sentences.** `test_extract_date_returns_iso_format` tells
   you exactly what broke when it fails.

4. **Use `tmp_path` for all file I/O.** Never write to real paths in tests.
   Never leave artifacts behind.

5. **Never mock what you own.** Stub external APIs (Groq, OpenAI, etc.), not
   your own functions. Exception: mocking `get_api_instance` in CLI tests to
   avoid real API calls is correct — you're stubbing the external boundary.

6. **Keep tests fast.** Unit tests under 100ms each. If a test takes >1s,
   you're probably hitting the network or disk unnecessarily.

7. **Failing import = valid RED.** If the function doesn't exist yet, an
   `ImportError` counts as RED. Write the function signature to get past it,
   then make the logic test pass.

8. **Guard integration tests.** Always `pytest.skip()` when the required API
   key or tool (ffmpeg) is missing. Never let CI fail because of missing
   credentials.

9. **Use markers consistently.** Every test class or function should have
   `@pytest.mark.unit`, `@pytest.mark.integration`, or `@pytest.mark.requires_ffmpeg`.

10. **Avoid placeholder tests.** `assert True` is not a test. If you can't
    test it yet, use `@pytest.mark.skip(reason="...")` with a clear reason.


## Acceptance criteria approach

Acceptance tests verify end-to-end behavior from the user's perspective.
They live in `tests/acceptance/` (or run manually against `test/` fixtures).

### Structure

Each acceptance scenario tests a complete user workflow:

```python
# tests/acceptance/test_transcribe_workflow.py

@pytest.mark.integration
@pytest.mark.slow
class TestTranscribeWorkflow:
    """End-to-end: user transcribes a file and gets output."""

    def test_single_file_produces_text_output(self, sample_audio_file, api_keys, tmp_path):
        """User runs: transcribe audio.m4a --api groq --output text"""
        api_key = api_keys.get("groq")
        if not api_key:
            pytest.skip("No Groq API key")

        runner = CliRunner()
        result = runner.invoke(main, [
            "--api", "groq",
            "--output", "text",
            "--output-dir", str(tmp_path),
            str(sample_audio_file),
        ])

        assert result.exit_code == 0
        txt_files = list(tmp_path.glob("*.txt"))
        assert len(txt_files) == 1
        assert len(txt_files[0].read_text(encoding="utf-8")) > 0
```

### What acceptance tests cover

| Scenario                          | What it validates                           |
|-----------------------------------|---------------------------------------------|
| Single file transcription         | CLI -> API -> output file created           |
| Folder batch processing           | Recursion, multiple files, skip existing    |
| JSON reuse                        | Existing JSON skips API call                |
| DaVinci SRT output                | Pause markers, filler word formatting       |
| Multi-speaker diarization         | Speaker labels in output                    |
| Chunking for large files          | Split -> transcribe -> merge -> valid output|
| Error recovery                    | Invalid file, missing key, network failure  |
| Force re-transcription            | `--force` overwrites existing output        |

### Golden file comparison

For deterministic outputs (formatters, parsers), compare against golden files
in `tests/fixtures/expected_outputs/`:

```python
def test_srt_output_matches_golden(self, tmp_path):
    result = TranscriptionResult.from_file(FIXTURES_DIR / "sample_speech.json")
    output = tmp_path / "test.srt"
    create_srt_file(result, output, format_type="standard")

    expected = (EXPECTED_OUTPUTS_DIR / "sample_speech.srt").read_text(encoding="utf-8")
    actual = output.read_text(encoding="utf-8")
    assert actual == expected
```

For non-deterministic outputs (API responses), use `llm_compare_texts` or
`llm_compare_transcriptions` from `conftest.py`.


## Adding tests for new features

### New transcription API

1. Add parser test in `tests/unit/test_parsers.py`:
   ```python
   class TestParseNewAPIFormat:
       def test_parse_basic_response(self):
           ...
   ```
2. Add integration test in `tests/integration/test_api_integration_newapi.py`
3. Add expected output fixtures if the response format is unique
4. Add golden files to `tests/fixtures/expected_outputs/`

### New output format

1. Add formatter test in `tests/unit/test_formatters.py`:
   ```python
   class TestCreateNewFormat:
       def test_creates_valid_output(self, tmp_path):
           ...
   ```
2. Add golden file comparison test
3. Add CLI integration test for the new `--output` option

### New CLI option

1. Add test in `tests/integration/test_cli.py`:
   ```python
   def test_new_option(self, sample_audio_file):
       runner = CliRunner()
       with patch("audio_transcribe.cli.get_api_instance") as mock:
           ...
           result = runner.invoke(main, ["--new-option", "value", str(sample_audio_file)])
           assert result.exit_code == 0
   ```

### New audio processing feature

1. Add test in `tests/unit/test_audio_processing.py`
2. Mark with `@pytest.mark.requires_ffmpeg` if it needs ffmpeg
3. Use `AudioSegment.silent()` to create synthetic test audio — don't depend
   on external files for unit tests
