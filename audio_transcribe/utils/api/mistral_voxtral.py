"""
Mistral Voxtral API implementation.

Supports:
- Segment-level timestamps (not word-level)
- Auto-detects language (cannot specify language in request)
- Batch-only (no streaming)
"""
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from loguru import logger

from audio_transcribe.utils.parsers import TranscriptionResult, generate_words_from_text
from audio_transcribe.utils.api.base import TranscriptionAPI


class MistralVoxtralAPI(TranscriptionAPI):
    """Mistral Voxtral transcription (cloud API only)."""

    # Capability flags - Voxtral only supports segment timestamps
    supports_word_timestamps: bool = False  # Only segment level
    supports_segment_timestamps: bool = True  # Segment-level only
    supports_speaker_diarization: bool = False
    supports_srt_format: bool = False
    supported_output_formats: List[str] = ["json"]  # Returns JSON with segments

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Mistral Voxtral API.

        Args:
            api_key: API key for Mistral (if not provided, will try to load from environment)
        """
        super().__init__(api_key)
        self.api_name = "mistral"

        if not self.api_key:
            self.api_key = self.load_from_env("MISTRAL_API_KEY")
            if not self.api_key:
                logger.error("No Mistral API key found. Please set the MISTRAL_API_KEY environment variable.")
                self.client = None
                return

        try:
            from mistralai import Mistral

            masked_key = self.mask_api_key(self.api_key)
            logger.debug(f"Initialized Mistral client with API key: {masked_key}")

            self.client = Mistral(api_key=self.api_key)
        except ImportError:
            logger.error("Mistral package not found. Please install it: uv add mistralai")
            self.client = None

    def list_models(self) -> List[str]:
        """List available models for Mistral Voxtral API."""
        # Voxtral models - use the correct model names
        return [
            "voxtral-mini-2507",
            "pixtral-12b-2409"  # Also supports audio
        ]

    def check_api_key(self) -> bool:
        """Check if Mistral API key is valid."""
        if not self.api_key:
            logger.error("No Mistral API key provided")
            return False

        if not self.client:
            logger.error("Mistral client not initialized")
            return False

        try:
            # Try a simple API call to validate
            response = self.client.models.list()
            if response and response.data:
                logger.debug(f"Mistral API key valid. Available models: {len(response.data)}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to validate Mistral API key: {str(e)}")
            return False

    def transcribe(self, audio_path: Union[str, Path], **kwargs) -> TranscriptionResult:
        """
        Transcribe audio file using Mistral Voxtral API.

        Important constraints:
        - timestamp_granularities only supports "segment" (not word)
        - NOT compatible with specifying language in same request
        - Must request segment timestamps explicitly

        Args:
            audio_path: Path to the audio file
            **kwargs: Additional Mistral-specific parameters:
                - model: Model to use (default: mistralai/Voxtral-Small-24B-2507)
                - language: Language code (WARNING: will be ignored!)
                - original_path: Original source file path before conversion

        Returns:
            Standardized TranscriptionResult object
        """
        if not self.client:
            raise ValueError("Mistral client not initialized")

        if isinstance(audio_path, Path):
            audio_path = str(audio_path)

        model = kwargs.get("model", "voxtral-mini-2507")

        # Warn if language specified (Voxtral doesn't support it)
        language = kwargs.get("language")
        if language:
            logger.warning(
                f"Mistral Voxtral auto-detects language. "
                f"Requested language '{language}' will be ignored."
            )
            # Remove language from kwargs so it doesn't get passed to API
            kwargs.pop("language", None)

        logger.info(f"Transcribing {audio_path} with Mistral Voxtral (model: {model})")

        try:
            # Prepare API parameters - Mistral SDK requires models.File with content
            from mistralai import models as mistral_models

            # Read file content
            with open(audio_path, "rb") as f:
                file_content = f.read()

            # Get MIME type
            mime_type = self._get_mime_type(audio_path)

            # Create File object
            audio_file = mistral_models.File(
                file_name=Path(audio_path).name,
                content=file_content,
                content_type=mime_type
            )

            # Request segment-level timestamps explicitly
            # Mistral only supports "segment" granularity (not "word")
            params = {
                "file": audio_file,
                "model": model,
                "timestamp_granularities": ["segment"]
            }

            # Call API with retry logic
            response = self.with_retry(
                lambda: self.client.audio.transcriptions.complete(**params)
            )

            # Extract data from response object
            if hasattr(response, "model_dump"):
                raw_data = response.model_dump()
            elif hasattr(response, "dict"):  # Pydantic v1
                raw_data = response.dict()
            elif hasattr(response, "__dict__"):
                raw_data = response.__dict__.copy()
            else:
                # Fallback to basic structure
                raw_data = {
                    "text": str(response),
                    "segments": []
                }

            # Add API name and model
            raw_data["api_name"] = self.api_name
            raw_data["model"] = model

            # Save raw response
            original_path = kwargs.get('original_path')
            self.save_result(raw_data, audio_path, original_path=original_path)

            # Parse response - convert segments to approximate word timestamps
            return self._parse_voxtral_response(raw_data)

        except Exception as e:
            logger.error(f"Mistral Voxtral transcription failed: {str(e)}")
            raise

    def _parse_voxtral_response(self, raw_data: Dict[str, Any]) -> TranscriptionResult:
        """
        Parse Mistral Voxtral response with segment-level timestamps.

        Voxtral returns:
        {
            "text": "Full transcript",
            "language": "en",
            "duration": 120.5,
            "segments": [
                {"text": "Hello world", "start": 0.0, "end": 2.5}
            ]
        }

        We convert segments to approximate word-level timestamps for compatibility.

        Args:
            raw_data: Raw response data from API

        Returns:
            Parsed TranscriptionResult
        """
        text = raw_data.get("text", "")
        segments = raw_data.get("segments", [])
        language = raw_data.get("language", "auto")

        # Convert segment-level timestamps to approximate word-level
        words = self._convert_segments_to_words(segments)

        return TranscriptionResult(
            text=text,
            words=words,
            confidence=0.0,
            language=language,
            speakers=[],
            segments=segments,
            api_name=self.api_name
        )

    def _convert_segments_to_words(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert segment-level timestamps to approximate word-level timestamps.

        Voxtral returns:
        {
          "segments": [
            {"text": "Hello world", "start": 0.0, "end": 2.5}
          ]
        }

        We convert to:
        {
          "words": [
            {"text": "Hello", "start": 0.0, "end": 1.25},
            {"text": "world", "start": 1.25, "end": 2.5}
          ]
        }

        Args:
            segments: List of segment dictionaries with text, start, end

        Returns:
            List of word dictionaries with approximate timestamps
        """
        words = []

        for segment in segments:
            segment_text = segment.get("text", "")
            if not segment_text:
                continue

            segment_words = segment_text.split()
            if not segment_words:
                continue

            start = segment.get("start", 0)
            end = segment.get("end", start + 1)
            duration = end - start

            # Distribute segment duration evenly across words
            word_duration = duration / len(segment_words)

            for i, word in enumerate(segment_words):
                word_start = start + (i * word_duration)
                word_end = word_start + word_duration

                words.append({
                    "text": word,
                    "start": word_start,
                    "end": word_end,
                    "confidence": 0.9  # Default confidence (not provided by Voxtral)
                })

        return words

    def _get_mime_type(self, audio_path: Union[str, Path]) -> str:
        """
        Get MIME type for audio file.

        Args:
            audio_path: Path to the audio file

        Returns:
            MIME type string
        """
        path = Path(audio_path)
        ext = path.suffix.lower()

        mime_types = {
            '.wav': 'audio/wav',
            '.mp3': 'audio/mpeg',
            '.flac': 'audio/flac',
            '.m4a': 'audio/mp4',
            '.mp4': 'audio/mp4',
            '.webm': 'audio/webm',
            '.ogg': 'audio/ogg',
            '.oga': 'audio/ogg',
            '.opus': 'audio/opus',
        }
        return mime_types.get(ext, 'audio/mpeg')
