"""
Base class for transcription APIs.
"""
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from loguru import logger

from audio_transcribe.utils.parsers import TranscriptionResult

class TranscriptionAPI(ABC):
    """Base class for all transcription API implementations."""

    # Capability flags - override in subclasses
    supports_word_timestamps: bool = True  # Can provide precise word timings
    supports_segment_timestamps: bool = False  # Can provide segment timestamps
    supports_speaker_diarization: bool = False  # Can identify speakers
    supports_srt_format: bool = True  # API can return SRT directly
    supported_output_formats: List[str] = ["text", "json"]  # What API returns natively

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the transcription API.

        Args:
            api_key: API key for the service (if not provided, will try to load from environment)
        """
        self.api_key = api_key
        self.api_name = "base"
        self.max_retries = 3
        self.retry_delay = 5  # seconds
        
    @abstractmethod
    def transcribe(self, audio_path: Union[str, Path], **kwargs) -> TranscriptionResult:
        """
        Transcribe audio file using the API.
        
        Args:
            audio_path: Path to the audio file
            **kwargs: Additional API-specific parameters
            
        Returns:
            Standardized TranscriptionResult object
        """
        pass
        
    @abstractmethod
    def check_api_key(self) -> bool:
        """
        Check if the API key is valid.
        
        Returns:
            True if the API key is valid, False otherwise
        """
        pass
        
    def list_models(self) -> List[str]:
        """
        List available models for this API.

        Returns:
            List of model IDs available for use
        """
        return []

    def get_best_response_format(self, preferred_formats: List[str]) -> str:
        """
        Get the best response format supported by this API.

        Strategy: Always try verbose_json first, then downgrade.
        Can turn json → txt but never txt → json (and then SRT, etc).

        Args:
            preferred_formats: List of formats in order of preference
                              (e.g., ["verbose_json", "json", "text"])

        Returns:
            Best supported format
        """
        for fmt in preferred_formats:
            if fmt in self.supported_output_formats:
                return fmt
        # Fallback to least common denominator
        return "text" if "text" in self.supported_output_formats else self.supported_output_formats[0]

    def save_result(self, result: Union[TranscriptionResult, Dict[str, Any]], audio_path: Union[str, Path], original_path: Optional[Union[str, Path]] = None) -> str:
        """
        Save transcription result to a JSON file.

        Args:
            result: TranscriptionResult object or raw dictionary
            audio_path: Path to the audio file (may be temporary)
            original_path: Path to the original input file (if different from audio_path)

        Returns:
            Path to the saved JSON file
        """
        # Use original_path for directory and filename if provided
        file_path = Path(original_path) if original_path else Path(audio_path)
        file_dir = file_path.parent
        file_name = file_path.stem
        
        # Save with API-specific suffix
        json_path = file_dir / f"{file_name}_{self.api_name}.json"
        
        if isinstance(result, TranscriptionResult):
            result.save(json_path)
        elif isinstance(result, dict):
            import json
            try:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.error(f"Failed to save raw result to {json_path}: {e}")
                return ""
        else:
            logger.warning(f"Cannot save result of type {type(result)}")
            return ""
            
        logger.info(f"Saved transcription result to {json_path}")
        
        return str(json_path)
        
    @staticmethod
    def load_from_env(env_var_name: str) -> Optional[str]:
        """
        Load API key from environment variable.
        
        Args:
            env_var_name: Name of the environment variable
            
        Returns:
            API key if found, None otherwise
        """
        api_key = os.getenv(env_var_name)
        if not api_key:
            logger.warning(f"No API key found in environment variable: {env_var_name}")
            return None
        return api_key
        
    def mask_api_key(self, api_key: str) -> str:
        """
        Mask API key for logging.
        
        Args:
            api_key: The API key to mask
            
        Returns:
            Masked API key (e.g., "sk-...1234")
        """
        if not api_key:
            return "None"
        if len(api_key) < 8:
            return "***"
        return f"{api_key[:3]}...{api_key[-4:]}"
        
    def mask_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """
        Create a copy of headers with masked sensitive values.
        
        Args:
            headers: Dictionary of headers
            
        Returns:
            Dictionary with masked values
        """
        masked = headers.copy()
        sensitive_keys = ['authorization', 'api-key', 'xi-api-key']
        
        for key in masked:
            if any(s in key.lower() for s in sensitive_keys):
                masked[key] = self.mask_api_key(masked[key])
                
        return masked
        
    def with_retry(self, func, *args, **kwargs):
        """
        Execute a function with retry logic.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function
        """
        retries = 0
        while retries < self.max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                retries += 1
                if retries >= self.max_retries:
                    logger.error(f"Failed after {self.max_retries} retries: {str(e)}")
                    raise
                logger.warning(f"Error: {str(e)}. Retrying in {self.retry_delay} seconds... (attempt {retries}/{self.max_retries})")
                time.sleep(self.retry_delay)
