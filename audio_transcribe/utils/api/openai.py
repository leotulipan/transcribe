"""
OpenAI API implementation.
"""
import os
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from loguru import logger

from audio_transcribe.utils.parsers import TranscriptionResult, parse_openai_format, generate_words_from_text
from audio_transcribe.utils.api.base import TranscriptionAPI
from audio_transcribe.utils.api.chunking import ChunkingMixin
from audio_transcribe.transcribe_helpers.audio_processing import convert_to_flac, get_api_file_size_limit
from audio_transcribe.transcribe_helpers.chunking import merge_transcripts

class OpenAIAPI(TranscriptionAPI, ChunkingMixin):
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
            
            # Log masked API key for debugging
            if self.api_key:
                masked_key = self.mask_api_key(self.api_key)
                logger.debug(f"Initializing OpenAI client with API key: {masked_key}")
                
            self.client = OpenAI(api_key=self.api_key)
            self.APIConnectionError = APIConnectionError
            self.AuthenticationError = AuthenticationError
            self.RateLimitError = RateLimitError
            self.APIError = APIError
        except ImportError:
            logger.error("OpenAI package not found. Please install it: uv add openai")
            self.client = None
            
    def list_models(self) -> List[str]:
        """
        List available models for OpenAI API.
        
        Returns:
            List of model IDs available for use
        """
        if not self.client:
            return []
            
        try:
            models = self.client.models.list()
            # Extract model IDs
            model_ids = [model.id for model in models.data]
            return model_ids
        except Exception as e:
            logger.error(f"Failed to list OpenAI models: {e}")
            return []

    def check_api_key(self) -> bool:
        """Check if OpenAI API key is valid."""
        if not self.api_key:
            logger.error("No OpenAI API key provided")
            return False
            
        if not self.client:
            logger.error("OpenAI client not initialized")
            return False
            
        try:
            # Use list_models to validate API key
            models = self.list_models()
            if models:
                logger.debug(f"OpenAI API key valid. Available models: {len(models)}")
                return True
            return False
        except Exception as e:
            if hasattr(self, 'AuthenticationError') and isinstance(e, self.AuthenticationError):
                logger.error("Invalid OpenAI API key")
            elif hasattr(self, 'APIConnectionError') and isinstance(e, self.APIConnectionError):
                logger.error(f"Connection error when validating OpenAI API key: {str(e)}")
            else:
                logger.error(f"Failed to validate OpenAI API key: {str(e)}")
            return False
    
    def transcribe_chunk(self, audio_chunk_path: Union[str, Path], 
                       chunk_start_ms: int = 0, **kwargs) -> Tuple[Dict[str, Any], int]:
        """
        Transcribe a single audio chunk with OpenAI.
        
        Args:
            audio_chunk_path: Path to audio chunk
            chunk_start_ms: Start time of chunk in milliseconds
            **kwargs: Additional arguments (model, language, etc.)
            
        Returns:
            Tuple of (transcription data, chunk_start_ms)
        """
        model = kwargs.get("model", "whisper-1")
        language = kwargs.get("language")
        
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
                raw_data = transcription_response.model_dump()
            elif hasattr(transcription_response, "__dict__"):
                raw_data = transcription_response.__dict__
            else:
                raw_data = {"text": str(transcription_response)}

            # Save the raw response for this chunk
            chunk_file_path = Path(audio_chunk_path)
            raw_chunk_json_path = chunk_file_path.with_name(f"{chunk_file_path.stem}_{self.api_name}.json")
            try:
                with open(raw_chunk_json_path, 'w', encoding='utf-8') as f_raw_chunk:
                    json.dump(raw_data, f_raw_chunk, indent=2, ensure_ascii=False)
                logger.info(f"Saved raw OpenAI chunk response to {raw_chunk_json_path}")
            except Exception as e_raw_chunk_save:
                logger.error(f"Failed to save raw OpenAI chunk response: {e_raw_chunk_save}")
                
            return raw_data, chunk_start_ms

    def merge_chunk_results(self, results: List[Tuple[Dict[str, Any], int]], **kwargs) -> TranscriptionResult:
        """
        Merge results from multiple chunks.
        
        Args:
            results: List of (chunk_result, start_ms) tuples
            **kwargs: Additional arguments (overlap, etc.)
            
        Returns:
            Merged TranscriptionResult
        """
        overlap = kwargs.get("overlap", 10)
        
        # Merge results from all chunks using the helper
        merged_data = merge_transcripts(results, overlap=overlap)
        
        # Add API name to merged data
        merged_data["api_name"] = self.api_name
        
        # Parse using our parser
        try:
            result = parse_openai_format(merged_data)
            return result
        except Exception as parse_err:
            logger.error(f"Failed to parse merged OpenAI response: {parse_err}")
            # Fallback to minimal result
            text = merged_data.get("text", "")
            words = generate_words_from_text(text)
            result = TranscriptionResult(
                text=text,
                words=words,
                api_name=self.api_name
            )
            return result
            
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
        
        # Prepare parameters
        model = kwargs.get("model", "whisper-1")
        language = kwargs.get("language")
        keep_flac = kwargs.get("keep_flac", False)
        
        logger.info(f"Transcribing {audio_path} with OpenAI Whisper (model: {model})")
        
        # Step 1: Convert to FLAC first (OpenAI requires this format)
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
            
            # Step 3: If FLAC file size exceeds limit, use chunking
            limit_mb = get_api_file_size_limit("openai")
            if file_size_mb > limit_mb:
                logger.info(f"FLAC file size ({file_size_mb:.2f}MB) exceeds OpenAI's {limit_mb}MB limit, using chunking")
                
                return self.transcribe_with_chunking(
                    flac_path,
                    chunk_length=chunk_length,
                    overlap=overlap,
                    model=model,
                    language=language
                )
                
            else:
                # For smaller files, use regular transcription
                try:
                    # Use the same transcribe_chunk method but just once
                    # This ensures consistent behavior and response handling
                    raw_data, _ = self.transcribe_chunk(
                        flac_path, 
                        chunk_start_ms=0, 
                        model=model, 
                        language=language
                    )
                    
                    # Add API name to the data
                    raw_data["api_name"] = self.api_name
                    
                    # Save raw response
                    self.save_result(raw_data, audio_path)
                    
                    # Parse using our parser
                    try:
                        result = parse_openai_format(raw_data)
                        return result
                    except Exception as parse_err:
                        logger.error(f"Failed to parse OpenAI response: {parse_err}")
                        # Create minimal result based on raw data text
                        text = raw_data.get("text", "")
                        words = generate_words_from_text(text)
                        result = TranscriptionResult(
                            text=text,
                            words=words,
                            api_name=self.api_name
                        )
                        return result
                        
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
