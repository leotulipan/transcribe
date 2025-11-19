"""
ElevenLabs API implementation.
"""
import os
import time
import json
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from loguru import logger

from audio_transcribe.utils.parsers import TranscriptionResult, parse_elevenlabs_format
from audio_transcribe.utils.api.base import TranscriptionAPI
from audio_transcribe.transcribe_helpers.audio_processing import extract_audio_from_mp4, check_file_size, get_api_file_size_limit

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
            
        # Log masked API key for debugging
        if self.api_key:
            masked_key = self.mask_api_key(self.api_key)
            logger.debug(f"Initializing ElevenLabs client with API key: {masked_key}")
            
    def check_api_key(self) -> bool:
        """Check if ElevenLabs API key is valid."""
        if not self.api_key:
            logger.error("No ElevenLabs API key provided")
            return False
            
        headers = {"xi-api-key": self.api_key}
        try:
            # Simple check - try to get user info
            response = requests.get(f"{self.base_url}/user", headers=headers)
            if response.status_code == 200:
                return True
            else:
                logger.error(f"ElevenLabs API key validation failed: {response.status_code} {response.text}")
                return False
        except Exception as e:
            logger.error(f"Failed to validate ElevenLabs API key: {str(e)}")
            return False
            
    def transcribe(self, audio_path: Union[str, Path], model_id: str = "scribe_v1", **kwargs) -> TranscriptionResult:
        """
        Transcribe audio file using ElevenLabs.
        
        Args:
            audio_path: Path to the audio file
            model_id: Model ID to use (default: scribe_v1)
            **kwargs: Additional ElevenLabs-specific parameters:
                - language: Language code (optional)
                - diarize: Boolean (default: False)
                - num_speakers: Integer (optional)
                
        Returns:
            Standardized TranscriptionResult object
        """
        if not self.api_key:
            raise ValueError("ElevenLabs API key not found")
            
        # Convert Path to string if needed
        if isinstance(audio_path, Path):
            audio_path = str(audio_path)
            
        # Check if input is MP4/Video and extract audio if needed
        temp_audio_path = None
        processing_path = audio_path
        
        if audio_path.lower().endswith(('.mp4', '.mkv', '.mov', '.avi')):
            logger.info(f"Input is video file: {audio_path}")
            logger.info("Extracting audio for ElevenLabs API...")
            
            extracted_path = extract_audio_from_mp4(audio_path)
            if extracted_path:
                processing_path = extracted_path
                temp_audio_path = extracted_path
                logger.info(f"Successfully extracted audio to: {processing_path}")
                
                # Check file size of extracted audio
                file_size_mb = os.path.getsize(processing_path) / (1024 * 1024)
                logger.info(f"Extracted audio size: {file_size_mb:.2f} MB")
            else:
                logger.warning("Failed to extract audio, attempting to upload original file")
        
        # Check file size limit (approx 1GB for ElevenLabs Scribe)
        limit_mb = get_api_file_size_limit("elevenlabs")
        if not check_file_size(processing_path, limit_mb):
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
            raise ValueError(f"File size exceeds ElevenLabs limit of {limit_mb}MB")
            
        url = f"{self.base_url}/speech-to-text"
        
        # Prepare headers (don't set Content-Type, requests handles it for multipart)
        headers = {
            "xi-api-key": self.api_key
        }
        
        # Prepare form data
        data = {
            "model_id": model_id,
            "tag_audio_events": "true",  # Always enable audio events
            "timestamps_granularity": "word"  # Explicitly request word timestamps
        }
        
        # Handle language parameter
        # Map 'language' kwarg to 'language_code' for ElevenLabs
        language = kwargs.get("language")
        if language:
            data["language_code"] = language
            
        # Handle diarization parameters
        diarize = kwargs.get("diarize", False)
        if diarize:
            data["diarize"] = "true"
            
            num_speakers = kwargs.get("num_speakers")
            if num_speakers:
                # Validate range 1-32
                try:
                    num_speakers_int = int(num_speakers)
                    if 1 <= num_speakers_int <= 32:
                        data["num_speakers"] = str(num_speakers_int)
                    else:
                        logger.warning(f"num_speakers {num_speakers} out of range (1-32), ignoring")
                except ValueError:
                    logger.warning(f"Invalid num_speakers value: {num_speakers}, ignoring")
        
        # Log payload keys for debugging
        logger.debug(f"ElevenLabs request payload keys: {list(data.keys())}")
        
        def make_request():
            with open(processing_path, 'rb') as f:
                files = {
                    'file': (os.path.basename(processing_path), f, 'audio/mpeg')
                }
                
                # Mask headers for logging
                masked_headers = self.mask_headers(headers)
                logger.debug(f"Sending request to {url}")
                logger.debug(f"Headers: {masked_headers}")
                
                response = requests.post(url, headers=headers, data=data, files=files)
                
                if response.status_code != 200:
                    logger.error(f"Error {response.status_code}: {response.text}")
                    
                response.raise_for_status()
                return response.json()
                
        try:
            logger.info(f"Transcribing {processing_path} with ElevenLabs...")
            start_time = time.time()
            
            result = self.with_retry(make_request)
            
            duration = time.time() - start_time
            logger.info(f"Transcription completed in {duration:.2f}s")
            
            # Save raw result
            self.save_result(result, audio_path) # Use original path for naming
            
            # Parse and return standardized result
            return parse_elevenlabs_format(result)
            
        except Exception as e:
            logger.error(f"ElevenLabs transcription failed: {str(e)}")
            raise
        finally:
            # Clean up temporary file if created
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.unlink(temp_audio_path)
                    logger.info(f"Deleted temporary audio file: {temp_audio_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {temp_audio_path}: {e}")
