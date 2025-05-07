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

from .parsers import TranscriptionResult


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
        
        # Import here to avoid circular imports
        from utils.parsers import parse_assemblyai_format
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
        from utils.parsers import parse_elevenlabs_format
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
            # Simple check - try to list models
            self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1
            )
            return True
        except Exception as e:
            logger.error(f"Failed to validate Groq API key: {str(e)}")
            return False
            
    def transcribe_chunk(self, audio_chunk_path: Union[str, Path], 
                        chunk_start_ms: int = 0, model: str = "whisper-large-v3", 
                        language: Optional[str] = None) -> Tuple[Dict[str, Any], int]:
        """
        Transcribe an audio chunk using Groq.
        
        Args:
            audio_chunk_path: Path to the audio chunk
            chunk_start_ms: Start time of the chunk in milliseconds
            model: Model to use for transcription
            language: Language code (optional)
            
        Returns:
            Tuple of (result_dict, chunk_start_ms)
        """
        logger.info(f"Transcribing chunk starting at {chunk_start_ms}ms with Groq")
        
        # Open the audio file
        with open(audio_chunk_path, "rb") as audio_file:
            # Create a chat completion with the audio file
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Transcribe this audio{' in ' + language if language else ''} with timestamps for every word."},
                        {"type": "audio", "audio": audio_file}
                    ]
                }
            ]
            
            def make_request():
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.0  # Use 0 temperature for reliable transcription
                )
                return completion.choices[0].message.content
                
            response_text = self.with_retry(make_request)
            
        # Process the response - simple JSON extraction
        result = None
        try:
            # Find JSON in the response (usually the model returns JSON)
            json_start = response_text.find('{')
            json_end = response_text.rfind('}')
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end+1]
                result = json.loads(json_str)
            else:
                # If no JSON found, create a basic structure with just the text
                logger.warning("No JSON found in Groq response, creating basic structure")
                result = {"text": response_text, "words": []}
        except Exception as e:
            logger.error(f"Error parsing Groq response: {str(e)}")
            result = {"text": response_text, "words": []}
            
        # Add API name to the result
        result["api_name"] = self.api_name
        
        return result, chunk_start_ms
            
    def transcribe(self, audio_path: Union[str, Path], **kwargs) -> TranscriptionResult:
        """
        Transcribe audio file using Groq.
        
        Args:
            audio_path: Path to the audio file
            **kwargs: Additional Groq-specific parameters:
                - language: Language code
                - model: Model to use (default: whisper-large-v3)
                - chunk_length: Length of each chunk in seconds (default: 600)
                - overlap: Overlap between chunks in seconds (default: 10)
                
        Returns:
            Standardized TranscriptionResult object
        """
        if not self.client:
            raise ValueError("Groq client not initialized")
            
        # Extract parameters
        language = kwargs.get("language")
        model = kwargs.get("model", "whisper-large-v3")
        chunk_length = kwargs.get("chunk_length", 600)  # seconds
        overlap = kwargs.get("overlap", 10)  # seconds
        
        logger.info(f"Transcribing {audio_path} with Groq (model: {model})")
        
        # Check if we need to chunk the audio
        from transcribe_helpers.audio_processing import check_audio_length
        audio_duration = check_audio_length(audio_path)
        
        if audio_duration <= chunk_length:
            # Single chunk transcription
            result_dict, _ = self.transcribe_chunk(audio_path, 0, model, language)
        else:
            # Multi-chunk transcription
            logger.info(f"Audio is {audio_duration}s long, splitting into chunks of {chunk_length}s with {overlap}s overlap")
            
            # Import chunking functions
            from transcribe_helpers.chunking import split_audio_file
            from transcribe_helpers.text_processing import find_longest_common_sequence
            
            # Split audio into chunks
            chunk_files = split_audio_file(
                audio_path, 
                chunk_length=chunk_length, 
                overlap=overlap
            )
            
            # Transcribe each chunk
            results = []
            for i, (chunk_file, chunk_start_ms) in enumerate(chunk_files):
                logger.info(f"Transcribing chunk {i+1}/{len(chunk_files)}")
                chunk_result, _ = self.transcribe_chunk(chunk_file, chunk_start_ms, model, language)
                results.append((chunk_result, chunk_start_ms))
                
            # Merge results
            def merge_transcripts(results: List[Tuple[Dict[str, Any], int]]) -> Dict[str, Any]:
                """Merge transcription chunks."""
                has_words = False
                words = []
                
                for chunk, chunk_start_ms in results:
                    # Process word timestamps if available
                    if "words" in chunk and chunk["words"]:
                        has_words = True
                        chunk_words = chunk["words"]
                        for word in chunk_words:
                            # Adjust word timestamps based on chunk start time
                            word["start"] = word["start"] + (chunk_start_ms / 1000)
                            word["end"] = word["end"] + (chunk_start_ms / 1000)
                        words.extend(chunk_words)
                
                # If no words, handle other response formats
                if not has_words:
                    texts = []
                    for chunk, _ in results:
                        text = chunk.get("text", "")
                        texts.append(text)
                    
                    merged_text = find_longest_common_sequence(texts)
                    return {"text": merged_text, "api_name": "groq"}
                
                # Sort words by start time
                words.sort(key=lambda x: x["start"])
                
                # Create merged text from words
                text = " ".join(word.get("text", "") for word in words if word.get("type", "") != "spacing")
                
                return {
                    "text": text,
                    "words": words,
                    "api_name": "groq"
                }
                
            result_dict = merge_transcripts(results)
        
        # Import here to avoid circular imports
        from utils.parsers import parse_groq_format
        result = parse_groq_format(result_dict)
        
        # Save result
        self.save_result(result, audio_path)
        
        return result


class OpenAIAPI(TranscriptionAPI):
    """OpenAI Whisper API implementation."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenAI API.
        
        Args:
            api_key: API key for OpenAI (if not provided, will try to load from environment)
        """
        super().__init__(api_key)
        self.api_name = "openai"
        
        if not self.api_key:
            self.api_key = self.load_from_env("OPENAI_API_KEY")
            
        # Import here to avoid circular imports
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        except ImportError:
            logger.error("OpenAI package not found. Please install it: uv add openai")
            self.client = None
            
    def check_api_key(self) -> bool:
        """Check if OpenAI API key is valid."""
        if not self.api_key:
            logger.error("No OpenAI API key provided")
            return False
            
        if not self.client:
            logger.error("OpenAI client not initialized")
            return False
            
        try:
            # Simple check - try to list models
            self.client.models.list()
            return True
        except Exception as e:
            logger.error(f"Failed to validate OpenAI API key: {str(e)}")
            return False
            
    def transcribe(self, audio_path: Union[str, Path], **kwargs) -> TranscriptionResult:
        """
        Transcribe audio file using OpenAI Whisper.
        
        Args:
            audio_path: Path to the audio file
            **kwargs: Additional OpenAI-specific parameters:
                - language: Language code
                - model: Whisper model to use (default: whisper-1)
                
        Returns:
            Standardized TranscriptionResult object
        """
        if not self.client:
            raise ValueError("OpenAI client not initialized")
            
        # Prepare parameters
        model = kwargs.get("model", "whisper-1")
        language = kwargs.get("language")
        response_format = kwargs.get("response_format", "verbose_json")
        
        logger.info(f"Transcribing {audio_path} with OpenAI Whisper")
        audio_file = open(audio_path, "rb")
        
        def make_request():
            params = {
                "model": model,
                "file": audio_file,
                "response_format": response_format
            }
            
            if language:
                params["language"] = language
                
            return self.client.audio.transcriptions.create(**params)
            
        # Use retry logic
        try:
            transcription = self.with_retry(make_request)
            logger.info("Transcription completed successfully")
            
            # Parse result
            data = {
                "text": transcription.text,
                "language": language or transcription.language,
                "model": model,
                "api_name": self.api_name,
            }
            
            # If we got words in verbose_json format
            if hasattr(transcription, "words") and transcription.words:
                data["words"] = transcription.words
                
            # Import here to avoid circular imports
            from utils.parsers import parse_openai_format
            result = parse_openai_format(data)
            
            # Save result
            self.save_result(result, audio_path)
            
            return result
            
        except Exception as e:
            logger.error(f"OpenAI transcription failed: {str(e)}")
            raise
        finally:
            audio_file.close()


def get_api_instance(api_name: str, api_key: Optional[str] = None) -> TranscriptionAPI:
    """
    Get an instance of the appropriate API class.
    
    Args:
        api_name: Name of the API to use
        api_key: API key to use (if None, will try to load from environment)
        
    Returns:
        Instance of a TranscriptionAPI subclass
    """
    api_name = api_name.lower()
    
    if api_name == "assemblyai":
        return AssemblyAIAPI(api_key)
    elif api_name == "elevenlabs":
        return ElevenLabsAPI(api_key)
    elif api_name == "groq":
        return GroqAPI(api_key)
    elif api_name == "openai":
        return OpenAIAPI(api_key)
    else:
        raise ValueError(f"Unknown API: {api_name}") 