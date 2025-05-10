"""
Unified transcription API classes for consistent access to various transcription services.
"""
import os
import time
import json
import base64
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
            # Try the new environment variable first
            self.api_key = self.load_from_env("ASSEMBLYAI_API_KEY")
            
            # If still not found, try the old environment variable for backward compatibility
            if not self.api_key:
                self.api_key = self.load_from_env("ASSEMBLY_AI_KEY")
            
        # Import here to avoid circular imports
        try:
            import assemblyai as aai
            self.aai = aai
            
            # Set the API key in settings instead of creating client
            if self.api_key:
                self.aai.settings.api_key = self.api_key
                self.client = True  # Just a flag to indicate we have a working setup
            else:
                self.client = None
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
            # Create a transcriber instance to verify API key
            transcriber = self.aai.Transcriber()
            return True
        except Exception as e:
            logger.error(f"Failed to validate AssemblyAI API key: {str(e)}")
            return False
            
    def transcribe(self, audio_path: Union[str, Path], **kwargs) -> Optional[Dict[str, Any]]:
        """
        Transcribe audio file using AssemblyAI.
        Returns the raw JSON response data as a dictionary, or None on failure.
        
        Args:
            audio_path: Path to the audio file
            **kwargs: Additional AssemblyAI-specific parameters:
                - language: Language code
                - speaker_labels: Enable speaker diarization
                - dual_channel: Enable dual channel transcription
                - model: Speech model to use (default, nano, small, medium, large, auto, best)
                
        Returns:
            Raw JSON response as dictionary, or None on failure
        """
        if not self.client:
            raise ValueError("AssemblyAI client not initialized")
        
        # Get the speech model from kwargs, default to "best"
        speech_model = kwargs.get("model", "best")
        
        # Validate the model
        valid_models = ["default", "nano", "small", "medium", "large", "auto", "best"]
        if speech_model not in valid_models:
            logger.warning(f"Invalid AssemblyAI model: {speech_model}, falling back to 'best'")
            speech_model = "best"
            
        logger.info(f"Using AssemblyAI model: {speech_model}")
        
        # Check if a language is specified
        language_code = kwargs.get("language")
        language_detection = True if language_code is None else False
        
        # Prepare transcription config
        config = self.aai.TranscriptionConfig(
            language_code=language_code,
            language_detection=language_detection,
            speaker_labels=kwargs.get("speaker_labels", True),
            dual_channel=kwargs.get("dual_channel", False),
            speech_model=speech_model,
            disfluencies=True  # Always enable disfluencies
        )
        
        # Convert Path to string if needed
        if isinstance(audio_path, Path):
            audio_path = str(audio_path)
            
        logger.info(f"Transcribing {audio_path} with AssemblyAI")
        
        # Create transcriber
        transcriber = self.aai.Transcriber()
        
        # Instead of passing the path directly, manually handle file upload
        try:
            with open(audio_path, "rb") as audio_file:
                # Submit and wait for completion using file content
                transcript = transcriber.transcribe(audio_file, config=config)
                
                logger.info(f"Transcription completed: {transcript.id}")
                
                # Return the raw JSON response
                result_dict = transcript.json_response
                
                # Save the raw response for debugging and reference
                file_dir = os.path.dirname(audio_path) if isinstance(audio_path, str) else audio_path.parent
                file_name = os.path.splitext(os.path.basename(audio_path))[0] if isinstance(audio_path, str) else audio_path.stem
                raw_json_path = os.path.join(file_dir, f"{file_name}.json") 
                try:
                    with open(raw_json_path, 'w', encoding='utf-8') as f:
                        json.dump(result_dict, f, indent=2, ensure_ascii=False)
                    logger.info(f"Saved raw AssemblyAI response to {raw_json_path}")
                except Exception as save_err:
                    logger.error(f"Failed to save raw AssemblyAI response: {save_err}")
                
                return result_dict
        except Exception as e:
            logger.error(f"Failed to transcribe with AssemblyAI: {str(e)}")
            # raise # Optionally re-raise
            return None


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
        
        # Convert Path to string if needed
        if isinstance(audio_path, Path):
            audio_path = str(audio_path)
            
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
        Transcribe a single audio chunk using Groq's audio transcriptions API.
        
        Args:
            audio_chunk_path: Path to the audio chunk file
            chunk_start_ms: Start time of this chunk in milliseconds
            model: Whisper model to use (default: whisper-large-v3)
            language: Language code (optional)
            
        Returns:
            Tuple of (chunk_result, chunk_start_ms)
        """
        if not self.client:
            raise ValueError("Groq client not initialized")
            
        # Convert Path to string if needed
        if isinstance(audio_chunk_path, Path):
            audio_chunk_path = str(audio_chunk_path)
            
        logger.info(f"Transcribing chunk starting at {chunk_start_ms}ms with Groq")
        
        # Open the file in binary mode
        with open(audio_chunk_path, "rb") as audio_file:
            try:
                # Use the audio.transcriptions.create endpoint
                # Pass the file directly as a tuple (filename, fileobj, content_type)
                start_time = time.time()
                result = self.client.audio.transcriptions.create(
                    file=("chunk.flac", audio_file, "audio/flac"),
                    model=model,
                    language=language,
                    response_format="verbose_json",
                    temperature=0,  # For best transcription quality
                    timestamp_granularities=["segment"]  # Get segment-level timestamps
                )
                transcription_time = time.time() - start_time
                logger.info(f"Chunk processed in {transcription_time:.2f}s")
                
                # Extract data from the result
                if hasattr(result, 'model_dump'):
                    # Handle Pydantic model response (newer Groq SDK)
                    data = result.model_dump()
                else:
                    # Handle dict-like response
                    data = dict(result)
                
                # Add API name to the result dict
                data["api_name"] = self.api_name
                
                return data, chunk_start_ms
                
            except Exception as e:
                logger.error(f"Error transcribing chunk with Groq: {str(e)}")
                raise
            
    def transcribe(self, audio_path: Union[str, Path], **kwargs) -> Optional[Dict[str, Any]]:
        """
        Transcribe audio file using Groq with proper chunking for long files.
        
        Args:
            audio_path: Path to the audio file
            **kwargs: Additional Groq-specific parameters:
                - language: Language code
                - model: Model to use (default: whisper-large-v3)
                - chunk_length: Length of each chunk in seconds (default: 600)
                - overlap: Overlap between chunks in seconds (default: 10)
                
        Returns:
            Dictionary with transcription results
        """
        if not self.client:
            raise ValueError("Groq client not initialized")
            
        # Extract parameters
        language = kwargs.get("language")
        model = kwargs.get("model", "whisper-large-v3")
        chunk_length = kwargs.get("chunk_length", 600)  # seconds
        overlap = kwargs.get("overlap", 10)  # seconds
        
        # Convert Path to string if needed
        if isinstance(audio_path, Path):
            audio_path = str(audio_path)
        
        logger.info(f"Transcribing {audio_path} with Groq (model: {model})")
        
        # Step 1: Convert to FLAC (always needed for Groq)
        from transcribe_helpers.audio_processing import convert_to_flac
        flac_path = convert_to_flac(audio_path)
        if not flac_path or not os.path.exists(flac_path):
            logger.error(f"Failed to convert audio to FLAC: {audio_path}")
            return None
            
        try:
            # Step 2: Check if we need to chunk the audio
            from pydub import AudioSegment
            audio = AudioSegment.from_file(flac_path)
            duration = len(audio)  # Duration in milliseconds
            logger.info(f"Audio duration: {duration/1000:.2f} seconds")
            
            # If audio is short enough, transcribe in a single request
            if duration <= chunk_length * 1000:
                logger.info("Audio is short enough for single transcription request")
                result_dict, _ = self.transcribe_chunk(flac_path, 0, model, language)
                
            else:
                # Step 3: Split audio into chunks and transcribe each
                logger.info(f"Audio is {duration/1000:.2f}s long, splitting into chunks of {chunk_length}s with {overlap}s overlap")
                
                chunk_ms = chunk_length * 1000
                overlap_ms = overlap * 1000
                total_chunks = (duration // (chunk_ms - overlap_ms)) + 1
                
                results = []
                
                for i in range(total_chunks):
                    start = i * (chunk_ms - overlap_ms)
                    end = min(start + chunk_ms, duration)
                    
                    logger.info(f"Processing chunk {i+1}/{total_chunks} ({start/1000:.1f}s - {end/1000:.1f}s)")
                    
                    # Extract chunk
                    chunk = audio[start:end]
                    
                    # Save chunk to temporary file
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.flac', delete=False) as temp_file:
                        chunk_path = temp_file.name
                        
                    chunk.export(chunk_path, format='flac')
                    
                    # Transcribe chunk
                    try:
                        result, _ = self.transcribe_chunk(chunk_path, start, model, language)
                        results.append((result, start))
                    finally:
                        # Clean up temp file
                        if os.path.exists(chunk_path):
                            os.unlink(chunk_path)
                
                # Step 4: Merge chunks
                result_dict = self._merge_transcripts(results, overlap)
                
            # Add API name to the response
            result_dict["api_name"] = self.api_name
            
            # Save raw result for debugging and reference
            file_dir = os.path.dirname(audio_path) if isinstance(audio_path, str) else audio_path.parent
            file_name = os.path.splitext(os.path.basename(audio_path))[0] if isinstance(audio_path, str) else audio_path.stem
            raw_json_path = os.path.join(file_dir, f"{file_name}.json")
            try:
                with open(raw_json_path, 'w', encoding='utf-8') as f:
                    json.dump(result_dict, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved raw Groq response to {raw_json_path}")
            except Exception as save_err:
                logger.error(f"Failed to save raw Groq response: {save_err}")
            
            # Import here to avoid circular imports
            from utils.parsers import parse_groq_format
            result = parse_groq_format(result_dict)
            
            # Save result
            result_path = self.save_result(result, audio_path)
            logger.info(f"Saved Groq transcription to {result_path}")
            
            return result_dict
            
        except Exception as e:
            logger.error(f"Error transcribing with Groq: {str(e)}")
            return None
        finally:
            # Clean up temporary FLAC file
            keep_flac = kwargs.get("keep_flac", False)
            if not keep_flac and flac_path and os.path.exists(flac_path) and os.path.basename(flac_path) != os.path.basename(audio_path):
                try:
                    os.unlink(flac_path)
                except:
                    pass
    
    def _merge_transcripts(self, results: List[Tuple[Dict[str, Any], int]], overlap: int = 10) -> Dict[str, Any]:
        """
        Merge transcription chunks, handling word-level timestamps.
        
        Args:
            results: List of (result, start_time_ms) tuples
            overlap: Overlap between chunks in seconds
            
        Returns:
            Merged transcription result
        """
        logger.info("Merging transcription chunks")
        
        # Initialize merged result
        merged_segments = []
        all_words = []
        
        # Process each chunk's segments and adjust timestamps
        overlap_sec = overlap  # Convert overlap to seconds
        
        for i, (chunk, chunk_start_ms) in enumerate(results):
            # Extract segments and words if available
            segments = chunk.get('segments', [])
            words = chunk.get('words', [])
            
            # Convert chunk_start_ms to seconds for timestamp adjustment
            chunk_start_sec = chunk_start_ms / 1000.0
            
            # Adjust segment timestamps
            for segment in segments:
                adjusted_segment = segment.copy()
                # For chunks after the first one, adjust by considering overlap
                if i > 0:
                    adjusted_segment['start'] = segment.get('start', 0) + chunk_start_sec - (i * overlap_sec)
                    adjusted_segment['end'] = segment.get('end', 0) + chunk_start_sec - (i * overlap_sec)
                else:
                    adjusted_segment['start'] = segment.get('start', 0) + chunk_start_sec
                    adjusted_segment['end'] = segment.get('end', 0) + chunk_start_sec
                merged_segments.append(adjusted_segment)
            
            # Adjust word timestamps
            adjusted_words = []
            for word in words:
                adjusted_word = word.copy()
                # For chunks after the first one, adjust by considering overlap
                if i > 0:
                    adjusted_word['start'] = word.get('start', 0) + chunk_start_sec - (i * overlap_sec)
                    adjusted_word['end'] = word.get('end', 0) + chunk_start_sec - (i * overlap_sec)
                else:
                    adjusted_word['start'] = word.get('start', 0) + chunk_start_sec
                    adjusted_word['end'] = word.get('end', 0) + chunk_start_sec
                adjusted_words.append(adjusted_word)
            
            all_words.extend(adjusted_words)
        
        # Sort words and segments by start time
        all_words.sort(key=lambda x: x.get('start', 0))
        merged_segments.sort(key=lambda x: x.get('start', 0))
        
        # Create final text from words (ensures proper ordering)
        full_text = " ".join(word.get('text', '') for word in all_words 
                           if word.get('type', '') != 'spacing')
        
        return {
            "text": full_text,
            "segments": merged_segments,
            "words": all_words,
            "api_name": "groq"
        }


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
            # Try to load from environment
            self.api_key = self.load_from_env("OPENAI_API_KEY")
            if not self.api_key:
                logger.error("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")
                self.client = None
                return
        
        # Import here to avoid circular imports
        try:
            from openai import OpenAI, APIConnectionError, AuthenticationError, RateLimitError, APIError
            self.client = OpenAI(api_key=self.api_key)
            self.APIConnectionError = APIConnectionError
            self.AuthenticationError = AuthenticationError
            self.RateLimitError = RateLimitError
            self.APIError = APIError
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
            # Simple check - try to list models without limit parameter
            self.client.models.list()
            logger.info("OpenAI API key is valid")
            return True
        except Exception as e:
            if hasattr(self, 'AuthenticationError') and isinstance(e, self.AuthenticationError):
                logger.error("Invalid OpenAI API key")
            elif hasattr(self, 'APIConnectionError') and isinstance(e, self.APIConnectionError):
                logger.error(f"Connection error when validating OpenAI API key: {str(e)}")
            else:
                logger.error(f"Failed to validate OpenAI API key: {str(e)}")
            return False
    
    def transcribe_chunk(self, audio_chunk_path: Union[str, Path], 
                       chunk_start_ms: int = 0, model: str = "whisper-1", 
                       language: Optional[str] = None) -> Tuple[Dict[str, Any], int]:
        """
        Transcribe a single audio chunk with OpenAI.
        
        Args:
            audio_chunk_path: Path to audio chunk
            chunk_start_ms: Start time of chunk in milliseconds
            model: OpenAI model to use
            language: Language code
            
        Returns:
            Tuple of (transcription data, chunk_start_ms)
        """
        with open(audio_chunk_path, "rb") as audio_file:
            params = {
                "model": model,
                "file": audio_file,
                "response_format": "verbose_json",
                "timestamp_granularities": ["word"]  # Enable word-level timestamps
            }
            
            if language:
                params["language"] = language
                
            transcription_response = self.with_retry(lambda: self.client.audio.transcriptions.create(**params))
            
            # Extract data from response object
            if hasattr(transcription_response, "model_dump"):
                data = transcription_response.model_dump()
            elif hasattr(transcription_response, "__dict__"):
                data = transcription_response.__dict__
            else:
                data = {"text": str(transcription_response)}
                
            return data, chunk_start_ms
            
    def transcribe(self, audio_path: Union[str, Path], **kwargs) -> TranscriptionResult:
        """
        Transcribe audio file using OpenAI Whisper.
        
        Args:
            audio_path: Path to the audio file
            **kwargs: Additional OpenAI-specific parameters:
                - language: Language code
                - model: Whisper model to use (default: whisper-1)
                - original_path: Original source file path before conversion
                - chunk_length: Length of each chunk in seconds (default: 500)
                - overlap: Overlap between chunks in seconds (default: 5)
                
        Returns:
            Standardized TranscriptionResult object
        """
        if not self.client:
            raise ValueError("OpenAI client not initialized")
            
        # Convert Path to string if needed
        if isinstance(audio_path, Path):
            audio_path = str(audio_path)
            
        # Get original path if present (for temporary files)
        original_path = kwargs.get("original_path", audio_path)
        file_path = Path(original_path)
        file_dir = file_path.parent
        file_name = file_path.stem
        
        # Prepare parameters
        model = kwargs.get("model", "whisper-1")
        language = kwargs.get("language")
        keep_flac = kwargs.get("keep_flac", False)
        
        logger.info(f"Transcribing {audio_path} with OpenAI Whisper (model: {model})")
        
        # Step 1: Convert to FLAC first (OpenAI requires this format)
        from transcribe_helpers.audio_processing import convert_to_flac
        is_converted = False
        flac_path = audio_path
        
        # Only convert if not already a FLAC file
        if not audio_path.lower().endswith('.flac'):
            logger.info(f"Converting input to FLAC format (required for OpenAI Whisper API)")
            flac_path = convert_to_flac(audio_path)
            if not flac_path:
                logger.error(f"Failed to convert audio file to FLAC format. Skipping file.")
                raise ValueError(f"Failed to convert audio file to FLAC format: {audio_path}")
            is_converted = True
            logger.info(f"Converted to FLAC: {flac_path}")
        
        try:
            # Step 2: Check FLAC file size (OpenAI max size is 25MB)
            file_size_mb = os.path.getsize(flac_path) / (1024 * 1024)
            logger.info(f"FLAC file size: {file_size_mb:.2f}MB")
            
            # Parameters for chunking large files
            chunk_length = kwargs.get("chunk_length", 500)  # Default 500 seconds (just under 25MB for most audio)
            overlap = kwargs.get("overlap", 5)             # Default 5 seconds overlap
            
            # Step 3: If FLAC file size exceeds 25MB, use chunking
            if file_size_mb > 25:
                logger.info(f"FLAC file size ({file_size_mb:.2f}MB) exceeds OpenAI's 25MB limit, using chunking")
                
                # Import chunking functionality
                from transcribe_helpers.chunking import split_audio, merge_transcripts
                
                chunks = split_audio(flac_path, chunk_length=chunk_length, overlap=overlap)
                results = []
                
                for i, (chunk_path, start_time) in enumerate(chunks):
                    logger.info(f"Transcribing chunk {i+1}/{len(chunks)}")
                    
                    try:
                        # Transcribe this chunk
                        chunk_result, _ = self.transcribe_chunk(
                            chunk_path, 
                            chunk_start_ms=start_time, 
                            model=model, 
                            language=language
                        )
                        results.append((chunk_result, start_time))
                    except Exception as e:
                        logger.error(f"Error transcribing chunk {i+1}: {str(e)}")
                        raise
                    finally:
                        # Clean up temporary chunk file
                        try:
                            if os.path.exists(chunk_path):
                                os.unlink(chunk_path)
                        except Exception as e:
                            logger.warning(f"Failed to delete temporary chunk file: {e}")
                
                # Merge results from all chunks
                merged_data = merge_transcripts(results, overlap=overlap)
                
                # Save raw merged data for analysis
                raw_json_path = file_dir / f"{file_name}.json"
                try:
                    with open(raw_json_path, 'w', encoding='utf-8') as f:
                        json.dump(merged_data, f, indent=2, ensure_ascii=False)
                    logger.info(f"Saved merged OpenAI response to {raw_json_path}")
                except Exception as save_err:
                    logger.error(f"Failed to save merged OpenAI response: {save_err}")
                    
                # Add API name to merged data
                merged_data["api_name"] = self.api_name
                
                # Parse using our parser
                try:
                    from utils.parsers import parse_openai_format
                    result = parse_openai_format(merged_data)
                    # Save the API-specific result
                    api_json_path = file_dir / f"{file_name}_{self.api_name}.json"
                    result.save(api_json_path)
                    logger.info(f"Saved standardized OpenAI result to {api_json_path}")
                    return result
                except Exception as parse_err:
                    logger.error(f"Failed to parse merged OpenAI response: {parse_err}")
                    # Fallback to minimal result
                    from utils.parsers import TranscriptionResult, generate_words_from_text
                    text = merged_data.get("text", "")
                    words = generate_words_from_text(text)
                    result = TranscriptionResult(
                        text=text,
                        words=words,
                        api_name=self.api_name
                    )
                    # Save the API-specific result
                    api_json_path = file_dir / f"{file_name}_{self.api_name}.json"
                    result.save(api_json_path)
                    logger.info(f"Saved minimal OpenAI result to {api_json_path}")
                    return result
                
            else:
                # For smaller files, use regular transcription
                try:
                    with open(flac_path, "rb") as audio_file:
                        def make_request():
                            params = {
                                "model": model,
                                "file": audio_file,
                                "response_format": "verbose_json",
                                "timestamp_granularities": ["word"]  # Enable word-level timestamps
                            }
                            
                            if language:
                                params["language"] = language
                                
                            return self.client.audio.transcriptions.create(**params)
                        
                        try:
                            # Execute the transcription request and get raw response
                            transcription_response = self.with_retry(make_request)
                            logger.info("OpenAI transcription API call completed")
                            
                            # Extract data from response object
                            if hasattr(transcription_response, "model_dump"):
                                data = transcription_response.model_dump()
                            elif hasattr(transcription_response, "__dict__"):
                                data = transcription_response.__dict__
                            else:
                                # Try direct string representation as fallback
                                try:
                                    data = {"text": str(transcription_response)}
                                except:
                                    data = {"text": "Failed to extract data from OpenAI response"}
                            
                            # Add API name to the data
                            data["api_name"] = self.api_name
                            
                            # Save raw response with standard name format
                            raw_json_path = file_dir / f"{file_name}.json"
                            try:
                                with open(raw_json_path, 'w', encoding='utf-8') as f:
                                    json.dump(data, f, indent=2, ensure_ascii=False)
                                logger.info(f"Saved raw OpenAI response to {raw_json_path}")
                            except Exception as save_err:
                                logger.error(f"Failed to save raw OpenAI response: {save_err}")
                            
                            # Continue with parsing
                            try:
                                from utils.parsers import parse_openai_format
                                result = parse_openai_format(data)
                                # Save the API-specific result
                                api_json_path = file_dir / f"{file_name}_{self.api_name}.json"
                                result.save(api_json_path)
                                logger.info(f"Saved standardized OpenAI result to {api_json_path}")
                                return result
                            except Exception as parse_err:
                                logger.error(f"Failed to parse OpenAI response: {parse_err}")
                                # Create minimal result based on raw data text
                                from utils.parsers import TranscriptionResult, generate_words_from_text
                                text = data.get("text", "")
                                words = generate_words_from_text(text)
                                result = TranscriptionResult(
                                    text=text,
                                    words=words,
                                    api_name=self.api_name
                                )
                                # Save the API-specific result
                                api_json_path = file_dir / f"{file_name}_{self.api_name}.json"
                                result.save(api_json_path)
                                logger.info(f"Saved minimal OpenAI result to {api_json_path}")
                                return result
                            
                        except Exception as e:
                            if hasattr(self, 'AuthenticationError') and isinstance(e, self.AuthenticationError):
                                logger.error("OpenAI authentication failed: Invalid API key")
                            elif hasattr(self, 'RateLimitError') and isinstance(e, self.RateLimitError):
                                logger.error("OpenAI rate limit exceeded. Please try again later.")
                            elif hasattr(self, 'APIError') and isinstance(e, self.APIError):
                                logger.error(f"OpenAI API error: {str(e)}")
                            else:
                                logger.error(f"OpenAI transcription failed: {str(e)}")
                            raise
                except Exception as e:
                    logger.error(f"OpenAI transcription failed: {str(e)}")
                    raise
        finally:
            # Clean up temporary FLAC file if we created it and don't want to keep it
            if is_converted and not keep_flac and flac_path and os.path.exists(flac_path):
                try:
                    os.unlink(flac_path)
                    logger.info(f"Deleted temporary FLAC file: {flac_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete temporary FLAC file: {e}")


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