"""
Base class for transcription API implementations.
"""
import os
import time
import json
import base64
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from loguru import logger

from utils.parsers import TranscriptionResult

class TranscriptionAPI(ABC):
    """Base class for all transcription API implementations."""
    
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
        
    def mask_api_key(self, key: str) -> str:
        """
        Mask an API key for safer logging.
        
        Args:
            key: The API key to mask
            
        Returns:
            A masked version of the API key showing only first 4 and last 4 characters
        """
        if not key or len(key) < 10:
            return "***"
        return f"{key[:4]}...{key[-4:]}"
        
    def mask_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """
        Create a copy of headers with API keys masked for safer logging.
        
        Args:
            headers: Dictionary of HTTP headers
            
        Returns:
            Dictionary with sensitive values masked
        """
        masked = headers.copy()
        sensitive_keys = ['authorization', 'api-key', 'xi-api-key']
        
        for k in masked:
            if k.lower() in sensitive_keys:
                masked[k] = self.mask_api_key(masked[k])
                
        return masked
        
    def load_from_env(self, env_var_name: str) -> Optional[str]:
        """
        Load API key from environment variable.
        
        Args:
            env_var_name: Name of the environment variable containing the API key
            
        Returns:
            API key from environment or None if not found
        """
        # Try getting from os.environ (loaded by python-dotenv in main script)
        api_key = os.environ.get(env_var_name)
        
        if api_key:
            logger.debug(f"Loaded {env_var_name} from environment")
            return api_key
            
        logger.warning(f"{env_var_name} not found in environment variables")
        return None
        
    def with_retry(self, func, max_retries: Optional[int] = None, retry_delay: Optional[int] = None):
        """
        Execute a function with retry logic.
        
        Args:
            func: Function to execute
            max_retries: Maximum number of retries (default: self.max_retries)
            retry_delay: Delay between retries in seconds (default: self.retry_delay)
            
        Returns:
            Result of the function
            
        Raises:
            Exception: If the function fails after all retries
        """
        retries = max_retries if max_retries is not None else self.max_retries
        delay = retry_delay if retry_delay is not None else self.retry_delay
        
        last_exception = None
        
        for attempt in range(retries + 1):
            try:
                return func()
            except Exception as e:
                last_exception = e
                if attempt < retries:
                    logger.warning(f"Attempt {attempt + 1}/{retries + 1} failed: {str(e)}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed. Last error: {str(e)}")
                    
        if last_exception:
            raise last_exception
            
    def save_result(self, result: Dict[str, Any], audio_path: Union[str, Path]) -> str:
        """
        Save transcription result to a JSON file.
        
        Args:
            result: Transcription result data
            audio_path: Path to the original audio file
            
        Returns:
            Path to the saved JSON file
        """
        if isinstance(audio_path, str):
            audio_path = Path(audio_path)
            
        # Create filename with API name suffix
        # e.g. audio_assemblyai.json
        file_name = f"{audio_path.stem}_{self.api_name}.json"
        output_path = audio_path.parent / file_name
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved raw transcription result to {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to save result to {output_path}: {str(e)}")
            return ""
