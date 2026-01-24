"""
OpenAI Extended API implementation with GPT-4o transcription models.

Supports:
- whisper-1: Original Whisper with word timestamps
- gpt-4o-transcribe: Text/json only, no timestamps
- gpt-4o-mini-transcribe: Text/json only, no timestamps
- gpt-4o-transcribe-diarize: Text/json with speaker diarization
"""
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from loguru import logger

from audio_transcribe.utils.parsers import TranscriptionResult, parse_openai_format, generate_words_from_text
from audio_transcribe.utils.api.base import TranscriptionAPI

# Model capability definitions
MODEL_CAPABILITIES = {
    "whisper-1": {
        "supports_word_timestamps": True,
        "supports_segment_timestamps": True,
        "supports_speaker_diarization": False,
        "supports_srt_format": True,
        "supported_output_formats": ["text", "json", "srt", "verbose_json", "vtt"]
    },
    "gpt-4o-transcribe": {
        "supports_word_timestamps": False,
        "supports_segment_timestamps": False,
        "supports_speaker_diarization": False,
        "supports_srt_format": False,
        "supported_output_formats": ["text", "json"]
    },
    "gpt-4o-mini-transcribe": {
        "supports_word_timestamps": False,
        "supports_segment_timestamps": False,
        "supports_speaker_diarization": False,
        "supports_srt_format": False,
        "supported_output_formats": ["text", "json"]
    },
    "gpt-4o-transcribe-diarize": {
        "supports_word_timestamps": False,
        "supports_segment_timestamps": False,
        "supports_speaker_diarization": True,
        "supports_srt_format": False,
        "supported_output_formats": ["text", "json"]
    }
}


class OpenAIExtendedAPI(TranscriptionAPI):
    """OpenAI transcription with GPT-4o models."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenAI Extended API.

        Args:
            api_key: API key for OpenAI (if not provided, will try to load from environment)
        """
        super().__init__(api_key)
        self.api_name = "openai_extended"

        if not self.api_key:
            self.api_key = self.load_from_env("OPENAI_API_KEY")
            if not self.api_key:
                logger.error("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")
                self.client = None
                return

        try:
            from openai import OpenAI

            if self.api_key:
                masked_key = self.mask_api_key(self.api_key)
                logger.debug(f"Initializing OpenAI Extended client with API key: {masked_key}")

            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            logger.error("OpenAI package not found. Please install it: uv add openai")
            self.client = None

    def list_models(self) -> List[str]:
        """List available models for OpenAI Extended API."""
        return list(MODEL_CAPABILITIES.keys())

    def check_api_key(self) -> bool:
        """Check if OpenAI API key is valid."""
        if not self.api_key:
            logger.error("No OpenAI API key provided")
            return False

        if not self.client:
            logger.error("OpenAI client not initialized")
            return False

        try:
            models = self.client.models.list()
            if models:
                logger.debug(f"OpenAI API key valid. Available models: {len(models.data)}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to validate OpenAI API key: {str(e)}")
            return False

    def _get_model_capabilities(self, model: str) -> Dict[str, Any]:
        """Get capabilities for a specific model."""
        return MODEL_CAPABILITIES.get(model, MODEL_CAPABILITIES["whisper-1"])

    def transcribe(self, audio_path: Union[str, Path], **kwargs) -> TranscriptionResult:
        """
        Transcribe audio file using OpenAI Extended models.

        Args:
            audio_path: Path to the audio file
            **kwargs: Additional OpenAI-specific parameters:
                - model: Model to use (default: gpt-4o-transcribe)
                - language: Language code
                - response_format: Output format (auto-detected based on model capabilities)
                - timestamp_granularities: Timestamp granularity (word or segment)
                - original_path: Original source file path before conversion

        Returns:
            Standardized TranscriptionResult object
        """
        if not self.client:
            raise ValueError("OpenAI client not initialized")

        if isinstance(audio_path, Path):
            audio_path = str(audio_path)

        original_path = kwargs.get("original_path", audio_path)
        file_path = Path(original_path)

        model = kwargs.get("model", "gpt-4o-transcribe")
        capabilities = self._get_model_capabilities(model)

        logger.info(f"Transcribing {audio_path} with OpenAI Extended (model: {model})")

        # Determine best response format based on model capabilities
        preferred_formats = kwargs.get("response_format", ["verbose_json", "json", "text"])
        if isinstance(preferred_formats, str):
            preferred_formats = [preferred_formats]

        # Use the model's supported formats to determine best format
        supported_formats = capabilities["supported_output_formats"]
        response_format = None
        for fmt in preferred_formats:
            if fmt in supported_formats:
                response_format = fmt
                break

        if not response_format:
            response_format = supported_formats[0]
            logger.warning(
                f"Requested format {preferred_formats} not supported by {model}. "
                f"Using {response_format} instead."
            )

        # Prepare API parameters
        with open(audio_path, "rb") as audio_file:
            params = {
                "model": model,
                "file": audio_file,
                "response_format": response_format
            }

            # Add language if specified
            language = kwargs.get("language")
            if language:
                params["language"] = language

            # Add timestamp granularities for models that support it
            if capabilities["supports_word_timestamps"] and response_format in ["verbose_json", "json"]:
                params["timestamp_granularities"] = ["word"]

            # Add granularities for segment timestamps if word not supported but segment is
            if capabilities["supports_segment_timestamps"] and response_format in ["verbose_json", "json"]:
                if not capabilities["supports_word_timestamps"]:
                    params["timestamp_granularities"] = ["segment"]

            try:
                # Call API with retry logic
                transcription_response = self.with_retry(
                    lambda: self.client.audio.transcriptions.create(**params)
                )

                # Extract data from response object
                if hasattr(transcription_response, "model_dump"):
                    raw_data = transcription_response.model_dump()
                elif hasattr(transcription_response, "dict"):  # Pydantic v1
                    raw_data = transcription_response.dict()
                elif hasattr(transcription_response, "__dict__"):
                    raw_data = transcription_response.__dict__.copy()
                else:
                    raw_data = {"text": str(transcription_response)}

                # Add API name and model
                raw_data["api_name"] = self.api_name
                raw_data["model"] = model

                # Save raw response
                self.save_result(raw_data, audio_path)

                # Parse response based on model capabilities
                return self._parse_response(raw_data, model, capabilities)

            except Exception as e:
                logger.error(f"OpenAI Extended transcription failed: {str(e)}")
                raise

    def _parse_response(
        self,
        raw_data: Dict[str, Any],
        model: str,
        capabilities: Dict[str, Any]
    ) -> TranscriptionResult:
        """
        Parse API response based on model capabilities.

        Args:
            raw_data: Raw response data from API
            model: Model name used for transcription
            capabilities: Model capabilities dictionary

        Returns:
            Parsed TranscriptionResult
        """
        text = raw_data.get("text", "")

        # Handle diarization model
        if capabilities["supports_speaker_diarization"]:
            return self._parse_diarization_response(raw_data, model)

        # Handle models with word timestamps
        if capabilities["supports_word_timestamps"] and "words" in raw_data:
            try:
                result = parse_openai_format(raw_data)
                return result
            except Exception as e:
                logger.warning(f"Failed to parse OpenAI format: {e}. Falling back to text-only.")

        # Handle models with segment timestamps
        if capabilities["supports_segment_timestamps"] and "segments" in raw_data:
            return self._parse_segment_response(raw_data, text)

        # Handle text-only models (generate approximate word timings)
        words = generate_words_from_text(text)

        return TranscriptionResult(
            text=text,
            words=words,
            confidence=0.0,
            language=raw_data.get("language", "auto"),
            speakers=[],
            segments=[],
            api_name=self.api_name
        )

    def _parse_diarization_response(self, raw_data: Dict[str, Any], model: str) -> TranscriptionResult:
        """
        Parse response from diarization model (gpt-4o-transcribe-diarize).

        The diarization model returns a different format with speaker information.
        Format:
        {
            "text": "Full transcript",
            "words": [
                {"text": "Hello", "start": 0.0, "end": 0.5, "speaker": "SPEAKER_1"},
                ...
            ]
        }
        """
        text = raw_data.get("text", "")
        words = raw_data.get("words", [])

        # Extract unique speakers
        speakers = sorted(
            set(w.get("speaker", "UNKNOWN") for w in words if w.get("speaker"))
        )

        # Build segments from words with speaker changes
        segments = []
        current_segment = None
        current_speaker = None

        for word in words:
            speaker = word.get("speaker", "UNKNOWN")
            text_content = word.get("text", "")

            if speaker != current_speaker:
                if current_segment:
                    segments.append(current_segment)
                current_segment = {
                    "text": text_content,
                    "start": word.get("start", 0),
                    "end": word.get("end", 0),
                    "speaker": speaker
                }
                current_speaker = speaker
            else:
                if current_segment:
                    current_segment["text"] += " " + text_content
                    current_segment["end"] = word.get("end", current_segment["end"])

        if current_segment:
            segments.append(current_segment)

        return TranscriptionResult(
            text=text,
            words=words,
            confidence=0.0,
            language=raw_data.get("language", "auto"),
            speakers=speakers,
            segments=segments,
            api_name=self.api_name
        )

    def _parse_segment_response(self, raw_data: Dict[str, Any], text: str) -> TranscriptionResult:
        """
        Parse response with segment-level timestamps.

        Converts segment timestamps to approximate word-level timestamps.
        """
        segments = raw_data.get("segments", [])

        # Distribute segment duration evenly across words
        words = []
        for segment in segments:
            segment_words = segment.get("text", "").split()
            if not segment_words:
                continue

            start = segment.get("start", 0)
            end = segment.get("end", start + 1)
            duration = end - start

            word_duration = duration / len(segment_words) if segment_words else duration

            for i, word in enumerate(segment_words):
                word_start = start + (i * word_duration)
                word_end = word_start + word_duration

                words.append({
                    "text": word,
                    "start": word_start,
                    "end": word_end,
                    "confidence": 0.9
                })

        return TranscriptionResult(
            text=text,
            words=words,
            confidence=0.0,
            language=raw_data.get("language", "auto"),
            speakers=[],
            segments=segments,
            api_name=self.api_name
        )
