"""
Unified transcription API classes for consistent access to various transcription services.
"""
import os
import time
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from loguru import logger

from audio_transcribe.utils.parsers import TranscriptionResult


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
        
    def save_result(self, result: TranscriptionResult, audio_path: Union[str, Path]) -> str:
        """
        Save transcription result to a JSON file.
        
        Args:
            result: TranscriptionResult object
            audio_path: Path to the original audio file
            
        Returns:
            Path to the saved JSON file
        """
        file_path = Path(audio_path)
        file_dir = file_path.parent
        file_name = file_path.stem
        
        # Save with API-specific suffix
        json_path = file_dir / f"{file_name}_{self.api_name}.json"
        result.save(json_path)
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


class AssemblyAIAPI(TranscriptionAPI):
    """AssemblyAI API implementation."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the AssemblyAI API.
        
        Args:
            api_key: API key for AssemblyAI (if not provided, will try to load from environment)
        """
        super().__init__(api_key)
        self.api_name = "assemblyai"
        
        if not self.api_key:
            self.api_key = self.load_from_env("ASSEMBLYAI_API_KEY")
            
        # Import here to avoid circular imports
        try:
            import assemblyai as aai
            self.aai = aai
            self.client = aai.Client(api_key=self.api_key) if self.api_key else None
        except ImportError:
            logger.error("AssemblyAI package not found. Please install it: uv add assemblyai")
            self.client = None
            
    def check_api_key(self) -> bool:
        """Check if AssemblyAI API key is valid."""
        if not self.api_key:
            logger.error("No AssemblyAI API key provided")
            return False
            
        if not self.client:
            logger.error("AssemblyAI client not initialized")
            return False
            
        try:
            # Simple check - try to get account info
            self.client.account.get_information()
            return True
        except Exception as e:
            logger.error(f"Failed to validate AssemblyAI API key: {str(e)}")
            return False
            
    def transcribe(self, audio_path: Union[str, Path], **kwargs) -> TranscriptionResult:
        """
        Transcribe audio file using AssemblyAI.
        
        Args:
            audio_path: Path to the audio file
            **kwargs: Additional AssemblyAI-specific parameters:
                - language: Language code
                - speaker_labels: Enable speaker diarization
                - dual_channel: Enable dual channel transcription
                
        Returns:
            Standardized TranscriptionResult object
        """
        if not self.client:
            raise ValueError("AssemblyAI client not initialized")
            
        # Prepare transcription config
        config = self.aai.TranscriptionConfig(
            language_code=kwargs.get("language", "en"),
            speaker_labels=kwargs.get("speaker_labels", True),
            dual_channel=kwargs.get("dual_channel", False)
        )
        
        logger.info(f"Transcribing {audio_path} with AssemblyAI")
        
        # Submit and wait for completion
        transcript = self.with_retry(
            self.client.transcribe, 
            audio_path, 
            config=config
        )
        
        logger.info(f"Transcription completed: {transcript.id}")
        
        # Convert to standardized format
        result_dict = transcript.json_response
        result_dict["api_name"] = self.api_name
        
        # Save raw JSON response for debugging and reference
        file_path = Path(audio_path)
        file_dir = file_path.parent
        file_name = file_path.stem
        raw_json_path = file_dir / f"{file_name}.json"
        try:
            with open(raw_json_path, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved raw AssemblyAI response to {raw_json_path}")
        except Exception as save_err:
            logger.error(f"Failed to save raw AssemblyAI response: {save_err}")
        
        # Import here to avoid circular imports
        from audio_transcribe.utils.parsers import parse_assemblyai_format
        result = parse_assemblyai_format(result_dict)
        
        # Save result
        self.save_result(result, audio_path)
        
        return result


class ElevenLabsAPI(TranscriptionAPI):
    """ElevenLabs API implementation."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the ElevenLabs API.
        
        Args:
            api_key: API key for ElevenLabs (if not provided, will try to load from environment)
        """
        super().__init__(api_key)
        self.api_name = "elevenlabs"
        self.base_url = "https://api.elevenlabs.io/v1"
        
        if not self.api_key:
            self.api_key = self.load_from_env("ELEVENLABS_API_KEY")
            
        # Import here to avoid circular imports
        try:
            import requests
            self.requests = requests
        except ImportError:
            logger.error("Requests package not found. Please install it: uv add requests")
            self.requests = None
            
    def check_api_key(self) -> bool:
        """Check if ElevenLabs API key is valid."""
        if not self.api_key:
            logger.error("No ElevenLabs API key provided")
            return False
            
        if not self.requests:
            logger.error("Requests package not initialized")
            return False
            
        try:
            # Try to get user info
            url = f"{self.base_url}/user"
            headers = {"xi-api-key": self.api_key}
            response = self.requests.get(url, headers=headers)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to validate ElevenLabs API key: {str(e)}")
            return False
            
    def transcribe(self, audio_path: Union[str, Path], **kwargs) -> TranscriptionResult:
        """
        Transcribe audio file using ElevenLabs.
        
        Args:
            audio_path: Path to the audio file
            **kwargs: Additional ElevenLabs-specific parameters:
                - language: Language code (optional)
                
        Returns:
            Standardized TranscriptionResult object
        """
        if not self.requests:
            raise ValueError("Requests package not initialized")
            
        logger.info(f"Transcribing {audio_path} with ElevenLabs")
        
        # Prepare request
        url = f"{self.base_url}/speech-to-text"
        headers = {"xi-api-key": self.api_key}
        
        # Prepare optional parameters
        data = {}
        if "language" in kwargs and kwargs["language"]:
            data["language"] = kwargs["language"]
            
        with open(audio_path, "rb") as f:
            files = {"file": f}
            
            # Make the API call with retry logic
            def make_request():
                response = self.requests.post(url, headers=headers, data=data, files=files)
                response.raise_for_status()
                return response.json()
                
            response_data = self.with_retry(make_request)
        
        # Add API name to the response
        response_data["api_name"] = self.api_name
        
        # Import here to avoid circular imports
        from audio_transcribe.utils.parsers import parse_elevenlabs_format
        result = parse_elevenlabs_format(response_data)
        
        # Save result
        self.save_result(result, audio_path)
        
        return result


class GroqAPI(TranscriptionAPI):
    """Groq API implementation for audio transcription."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Groq API.
        
        Args:
            api_key: API key for Groq (if not provided, will try to load from environment)
        """
        super().__init__(api_key)
        self.api_name = "groq"
        
        if not self.api_key:
            self.api_key = self.load_from_env("GROQ_API_KEY")
            
        # Import here to avoid circular imports
        try:
            from groq import Groq
            from groq.types.audio import AudioInput
            self.groq = Groq
            self.AudioInput = AudioInput
            self.client = Groq(api_key=self.api_key) if self.api_key else None
        except ImportError:
            logger.error("Groq package not found. Please install it: uv add groq")
            self.client = None
            
    def check_api_key(self) -> bool:
        """Check if Groq API key is valid."""
        if not self.api_key:
            logger.error("No Groq API key provided")
            return False
            
        if not self.client:
            logger.error("Groq client not initialized")
            return False
            
        try:
            # Simple test request
            self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1
            )
            return True
        except Exception as e:
            logger.error(f"Failed to validate Groq API key: {str(e)}")
            return False
            
    def transcribe(self, audio_path: Union[str, Path], **kwargs) -> TranscriptionResult:
        """
        Transcribe audio file using Groq.
        
        Args:
            audio_path: Path to the audio file
            **kwargs: Additional Groq-specific parameters:
                - language: Language code
                - model: Model to use (default: whisper-large-v3-turbo)
                
        Returns:
            Standardized TranscriptionResult object
        """
        if not self.client:
            raise ValueError("Groq client not initialized")
        
        # Extract parameters
        language = kwargs.get("language")
        model = kwargs.get("model", "whisper-large-v3-turbo")
        
        logger.info(f"Transcribing {audio_path} with Groq (model: {model})")
        
        # Open the audio file
        with open(audio_path, "rb") as audio_file:
            # Prepare the messages
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": f"Transcribe this audio{' in ' + language if language else ''}."},
                        {"type": "audio", "audio": audio_file}
                    ]
                }
            ]
            
            # Submit the transcription request
            response = self.with_retry(
                self.client.chat.completions.create,
                model=model,
                messages=messages,
                temperature=0
            )
            
            # Extract the transcription from the response
            transcription = response.choices[0].message.content
            
            # Basic parsing of the response (might be JSON or plain text)
            try:
                if transcription.startswith("{") and transcription.endswith("}"):
                    # Try to parse as JSON
                    import json
                    result_dict = json.loads(transcription)
                else:
                    # Create a basic structure with just the text
                    result_dict = {"text": transcription, "words": []}
            except Exception as e:
                logger.error(f"Error parsing Groq response: {str(e)}")
                result_dict = {"text": transcription, "words": []}
            
            # Add API name to the response
            result_dict["api_name"] = self.api_name
            
            # Save raw JSON response for debugging and reference
            file_path = Path(audio_path)
            file_dir = file_path.parent
            file_name = file_path.stem
            raw_json_path = file_dir / f"{file_name}.json"
            try:
                with open(raw_json_path, 'w', encoding='utf-8') as f:
                    json.dump(result_dict, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved raw Groq response to {raw_json_path}")
            except Exception as save_err:
                logger.error(f"Failed to save raw Groq response: {save_err}")
            
            # Import here to avoid circular imports
            from audio_transcribe.utils.parsers import parse_groq_format
            result = parse_groq_format(result_dict)
            
            # Save result
            self.save_result(result, audio_path)
            
            return result


def get_api_instance(api_name: str, api_key: Optional[str] = None) -> TranscriptionAPI:
    """
    Factory function to get the appropriate API instance.
    
    Args:
        api_name: Name of the API to use
        api_key: API key (optional, will try to load from environment if not provided)
        
    Returns:
        TranscriptionAPI instance
    """
    if api_name == "assemblyai":
        return AssemblyAIAPI(api_key)
    elif api_name == "elevenlabs":
        return ElevenLabsAPI(api_key)
    elif api_name == "groq":
        return GroqAPI(api_key)
    else:
        raise ValueError(f"Unknown API: {api_name}") 