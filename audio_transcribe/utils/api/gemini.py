"""
Google Gemini API implementation using generateContent API.

Supports:
- Inline audio for files ≤20MB
- Files API upload for larger files
- Text-only output (no timestamps)
"""
import base64
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from loguru import logger

from audio_transcribe.utils.parsers import TranscriptionResult, generate_words_from_text
from audio_transcribe.utils.api.base import TranscriptionAPI


class GeminiAPI(TranscriptionAPI):
    """Google Gemini transcription using generateContent API."""

    # Capability flags - Gemini is text-only
    supports_word_timestamps: bool = False
    supports_segment_timestamps: bool = False
    supports_speaker_diarization: bool = False
    supports_srt_format: bool = False
    supported_output_formats: List[str] = ["text"]

    # File size limit for inline audio (20MB)
    INLINE_SIZE_LIMIT = 20 * 1024 * 1024  # 20MB in bytes

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini API.

        Args:
            api_key: API key for Google Gemini (if not provided, will try to load from environment)
        """
        super().__init__(api_key)
        self.api_name = "gemini"

        if not self.api_key:
            self.api_key = self.load_from_env("GEMINI_API_KEY")
            if not self.api_key:
                logger.error("No Gemini API key found. Please set the GEMINI_API_KEY environment variable.")
                self.client = None
                return

        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self.genai = genai

            masked_key = self.mask_api_key(self.api_key)
            logger.debug(f"Initialized Gemini client with API key: {masked_key}")
        except ImportError:
            logger.error("google-generativeai package not found. Please install it: uv add google-generativeai")
            self.client = None
            self.genai = None

    def list_models(self) -> List[str]:
        """List available models for Gemini API."""
        if not self.genai:
            return []

        try:
            models = list(self.genai.list_models())
            # Filter for models that support audio
            audio_models = [
                m.name for m in models
                if "generateContent" in m.supported_generation_methods
                and "flash" in m.name.lower()
            ]
            return audio_models
        except Exception as e:
            logger.error(f"Failed to list Gemini models: {e}")
            return ["gemini-2.5-flash", "gemini-1.5-flash"]

    def check_api_key(self) -> bool:
        """Check if Gemini API key is valid."""
        if not self.api_key:
            logger.error("No Gemini API key provided")
            return False

        if not self.genai:
            logger.error("Gemini client not initialized")
            return False

        try:
            # Try to list models as a validation check
            models = self.list_models()
            if models:
                logger.debug(f"Gemini API key valid. Available models: {len(models)}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to validate Gemini API key: {str(e)}")
            return False

    def transcribe(self, audio_path: Union[str, Path], **kwargs) -> TranscriptionResult:
        """
        Transcribe audio file using Gemini generateContent API.

        Args:
            audio_path: Path to the audio file
            **kwargs: Additional Gemini-specific parameters:
                - model: Model to use (default: gemini-2.5-flash)
                - language: Language code (optional)
                - original_path: Original source file path before conversion

        Returns:
            Standardized TranscriptionResult object
        """
        if not self.genai:
            raise ValueError("Gemini client not initialized")

        if isinstance(audio_path, Path):
            audio_path = str(audio_path)

        model = kwargs.get("model", "gemini-2.5-flash")
        language = kwargs.get("language")

        logger.info(f"Transcribing {audio_path} with Gemini (model: {model})")

        # Check file size to determine method
        file_size = os.path.getsize(audio_path)

        original_path = kwargs.get("original_path")

        if file_size <= self.INLINE_SIZE_LIMIT:
            logger.debug(f"File size ({file_size / 1024 / 1024:.2f}MB) <= 20MB limit, using inline audio")
            return self._transcribe_inline(audio_path, model, language, original_path=original_path)
        else:
            logger.debug(f"File size ({file_size / 1024 / 1024:.2f}MB) > 20MB limit, using Files API")
            return self._transcribe_with_files_api(audio_path, model, language, original_path=original_path)

    def _transcribe_inline(self, audio_path: str, model: str, language: Optional[str], original_path: Optional[str] = None) -> TranscriptionResult:
        """
        Transcribe with inline audio (≤20MB).

        Args:
            audio_path: Path to audio file
            model: Gemini model name
            language: Language code (optional)

        Returns:
            TranscriptionResult
        """
        try:
            # Read and encode audio
            with open(audio_path, 'rb') as f:
                audio_data = base64.b64encode(f.read()).decode('utf-8')

            mime_type = self._get_mime_type(audio_path)

            # Build prompt
            prompt = "Generate a detailed transcript of this audio. Include punctuation and capitalize appropriately."
            if language:
                prompt += f" The audio is in {self._get_language_name(language)}."

            # Create model instance
            genai_model = self.genai.GenerativeModel(model)

            # Prepare content with inline audio
            content = [
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": audio_data
                    }
                },
                {"text": prompt}
            ]

            # Generate transcription with retry logic
            response = self.with_retry(
                lambda: genai_model.generate_content(content)
            )

            # Extract text from response
            text = response.text
            logger.debug(f"Received transcription: {len(text)} characters")

            # Save raw response
            raw_data = {
                "text": text,
                "model": model,
                "api_name": self.api_name,
                "method": "inline"
            }
            self.save_result(raw_data, audio_path, original_path=original_path)

            # Generate approximate word timings (no timestamps from Gemini)
            words = generate_words_from_text(text)

            return TranscriptionResult(
                text=text,
                words=words,
                confidence=0.0,
                language=language or "auto",
                speakers=[],
                segments=[],
                api_name=self.api_name
            )

        except Exception as e:
            logger.error(f"Gemini inline transcription failed: {str(e)}")
            raise

    def _transcribe_with_files_api(self, audio_path: str, model: str, language: Optional[str], original_path: Optional[str] = None) -> TranscriptionResult:
        """
        Transcribe with Files API upload (>20MB).

        Args:
            audio_path: Path to audio file
            model: Gemini model name
            language: Language code (optional)

        Returns:
            TranscriptionResult
        """
        try:
            # Upload file first
            file_uri = self._upload_file(audio_path)

            # Build prompt
            prompt = "Generate a detailed transcript of this audio. Include punctuation and capitalize appropriately."
            if language:
                prompt += f" The audio is in {self._get_language_name(language)}."

            # Create model instance
            genai_model = self.genai.GenerativeModel(model)

            # Prepare content with file reference
            mime_type = self._get_mime_type(audio_path)
            content = [
                {
                    "file_data": {
                        "file_uri": file_uri,
                        "mime_type": mime_type
                    }
                },
                {"text": prompt}
            ]

            # Generate transcription with retry logic
            response = self.with_retry(
                lambda: genai_model.generate_content(content)
            )

            # Extract text from response
            text = response.text
            logger.debug(f"Received transcription: {len(text)} characters")

            # Save raw response
            raw_data = {
                "text": text,
                "model": model,
                "api_name": self.api_name,
                "method": "files_api",
                "file_uri": file_uri
            }
            self.save_result(raw_data, audio_path, original_path=original_path)

            # Generate approximate word timings (no timestamps from Gemini)
            words = generate_words_from_text(text)

            return TranscriptionResult(
                text=text,
                words=words,
                confidence=0.0,
                language=language or "auto",
                speakers=[],
                segments=[],
                api_name=self.api_name
            )

        except Exception as e:
            logger.error(f"Gemini Files API transcription failed: {str(e)}")
            raise

    def _upload_file(self, audio_path: str) -> str:
        """
        Upload file to Gemini Files API.

        Args:
            audio_path: Path to audio file

        Returns:
            File URI for use in generateContent
        """
        try:
            # Upload file using Gemini's upload_file method
            file_display_name = Path(audio_path).name
            mime_type = self._get_mime_type(audio_path)

            logger.info(f"Uploading {file_display_name} to Gemini Files API...")

            uploaded_file = self.genai.upload_file(
                path=audio_path,
                display_name=file_display_name,
                mime_type=mime_type
            )

            logger.info(f"File uploaded successfully: {uploaded_file.name}")
            logger.debug(f"File URI: {uploaded_file.uri}")

            return uploaded_file.uri

        except Exception as e:
            logger.error(f"Failed to upload file to Gemini Files API: {str(e)}")
            raise

    def _get_mime_type(self, audio_path: str) -> str:
        """
        Get MIME type for audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            MIME type string
        """
        ext = Path(audio_path).suffix.lower()
        mime_types = {
            '.wav': 'audio/wav',
            '.mp3': 'audio/mp3',
            '.flac': 'audio/flac',
            '.m4a': 'audio/mp4',
            '.mp4': 'audio/mp4',
            '.webm': 'audio/webm',
            '.ogg': 'audio/ogg',
            '.oga': 'audio/ogg',
            '.opus': 'audio/opus',
            '.amr': 'audio/amr',
            '.awb': 'audio/amr-wb',
            '.3gp': 'audio/3gpp',
        }
        return mime_types.get(ext, 'audio/mpeg')

    def _get_language_name(self, language_code: str) -> str:
        """
        Convert language code to full language name.

        Args:
            language_code: ISO-639-1 or ISO-639-3 language code

        Returns:
            Full language name
        """
        language_names = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'ja': 'Japanese',
            'ko': 'Korean',
            'zh': 'Chinese',
            'ar': 'Arabic',
            'hi': 'Hindi',
            'nl': 'Dutch',
            'pl': 'Polish',
            'tr': 'Turkish',
            'sv': 'Swedish',
            'da': 'Danish',
            'no': 'Norwegian',
            'fi': 'Finnish',
        }
        return language_names.get(language_code.lower(), language_code)
