"""
AssemblyAI API implementation.
"""
import os
import time
import json
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from loguru import logger

from utils.parsers import TranscriptionResult, parse_assemblyai_format
from utils.api.base import TranscriptionAPI

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
        self.base_url = "https://api.assemblyai.com/v2"
        
        if not self.api_key:
            self.api_key = self.load_from_env("ASSEMBLYAI_API_KEY")
            
        # Log masked API key for debugging
        if self.api_key:
            masked_key = self.mask_api_key(self.api_key)
            logger.debug(f"Initializing AssemblyAI client with API key: {masked_key}")
            
    def check_api_key(self) -> bool:
        """Check if AssemblyAI API key is valid."""
        if not self.api_key:
            logger.error("No AssemblyAI API key provided")
            return False
            
        headers = {"authorization": self.api_key}
        try:
            # Simple check - try to list transcripts (limit 1)
            response = requests.get(f"{self.base_url}/transcript?limit=1", headers=headers)
            if response.status_code == 200:
                return True
            else:
                logger.error(f"AssemblyAI API key validation failed: {response.status_code} {response.text}")
                return False
        except Exception as e:
            logger.error(f"Failed to validate AssemblyAI API key: {str(e)}")
            return False
            
    def transcribe_chunk(self, audio_chunk_path: Union[str, Path], 
                        chunk_start_ms: int = 0, model: str = "best", 
                        language: Optional[str] = None) -> Tuple[Dict[str, Any], int]:
        """
        Transcribe a single audio chunk using AssemblyAI.
        
        Args:
            audio_chunk_path: Path to the audio chunk file
            chunk_start_ms: Start time of this chunk in milliseconds
            model: Model to use (default: best)
            language: Language code (optional)
            
        Returns:
            Tuple of (chunk_result, chunk_start_ms)
        """
        # AssemblyAI doesn't support direct chunk transcription in the same way as local models
        # But we can implement it by uploading and transcribing the chunk
        
        # This is a simplified implementation that reuses the main transcribe logic
        # In a real chunked implementation, we would need to handle the upload/transcribe flow for each chunk
        
        # For now, we'll raise NotImplementedError as AssemblyAI handles large files well natively
        # and doesn't strictly need client-side chunking like Groq might
        raise NotImplementedError("Client-side chunking not yet implemented for AssemblyAI")

    def transcribe(self, audio_path: Union[str, Path], **kwargs) -> TranscriptionResult:
        """
        Transcribe audio file using AssemblyAI.
        
        Args:
            audio_path: Path to the audio file
            **kwargs: Additional AssemblyAI-specific parameters:
                - language: Language code
                - speaker_labels: Boolean (default: True)
                - speakers_expected: Optional integer
                - disfluencies: Boolean (default: True) - Transcribe filler words
                - model: Model tier (nano, best, etc.) - default: best
                - chunk_length: Length of each chunk in seconds (default: 600)
                - overlap: Overlap between chunks in seconds (default: 10)
                
        Returns:
            TranscriptionResult object with the standardized format
        """
        if not self.api_key:
            raise ValueError("AssemblyAI API key not found")
            
        headers = {"authorization": self.api_key}
        
        # Step 1: Upload the file
        logger.info(f"Uploading {audio_path} to AssemblyAI...")
        
        def upload_file():
            with open(audio_path, "rb") as f:
                response = requests.post(
                    f"{self.base_url}/upload",
                    headers=headers,
                    data=f
                )
            response.raise_for_status()
            return response.json()
            
        try:
            upload_result = self.with_retry(upload_file)
            upload_url = upload_result["upload_url"]
            logger.info(f"Upload complete. URL: {upload_url}")
        except Exception as e:
            logger.error(f"Failed to upload file to AssemblyAI: {str(e)}")
            raise
            
        # Step 2: Submit for transcription
        logger.info("Submitting for transcription...")
        
        # Prepare parameters
        data = {
            "audio_url": upload_url,
            "speaker_labels": kwargs.get("speaker_labels", True),
            "disfluencies": kwargs.get("disfluencies", True), # Default to True as per requirements
            "language_detection": True # Default to auto-detect
        }
        
        # Handle language parameter
        language = kwargs.get("language")
        if language:
            data["language_code"] = language
            data["language_detection"] = False
            
        # Handle model parameter
        model = kwargs.get("model", "best")
        if model and model.lower() != "auto":
            # Map "whisper-large-v3" or similar to "best" if passed by mistake
            if "whisper" in model.lower():
                logger.warning(f"Model '{model}' not supported by AssemblyAI, falling back to 'best'")
                model = "best"
            data["speech_model"] = model
            
        # Handle expected speakers
        speakers_expected = kwargs.get("speakers_expected")
        if speakers_expected:
            data["speakers_expected"] = speakers_expected
            
        def submit_transcription():
            response = requests.post(
                f"{self.base_url}/transcript",
                json=data,
                headers=headers
            )
            response.raise_for_status()
            return response.json()
            
        try:
            transcription_submission = self.with_retry(submit_transcription)
            transcript_id = transcription_submission["id"]
            logger.info(f"Transcription submitted. ID: {transcript_id}")
        except Exception as e:
            logger.error(f"Failed to submit transcription: {str(e)}")
            raise
            
        # Step 3: Poll for completion
        logger.info("Waiting for transcription to complete...")
        
        while True:
            try:
                response = requests.get(
                    f"{self.base_url}/transcript/{transcript_id}",
                    headers=headers
                )
                response.raise_for_status()
                result = response.json()
                
                status = result["status"]
                
                if status == "completed":
                    logger.info("Transcription completed!")
                    break
                elif status == "error":
                    error_msg = result.get("error")
                    logger.error(f"Transcription failed: {error_msg}")
                    raise ValueError(f"AssemblyAI transcription failed: {error_msg}")
                else:
                    # Still processing
                    time.sleep(3)
            except Exception as e:
                # If it's a transient network error, we might want to continue polling
                # But for now, we'll raise it
                logger.error(f"Error polling status: {str(e)}")
                raise
                
        # Step 4: Save raw result
        self.save_result(result, audio_path)
        
        # Step 5: Parse and return standardized result
        return parse_assemblyai_format(result)
