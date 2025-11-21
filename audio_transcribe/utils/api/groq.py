"""
Groq API implementation.
"""
import os
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from loguru import logger
import tempfile

from audio_transcribe.utils.parsers import TranscriptionResult, parse_groq_format
from audio_transcribe.utils.api.base import TranscriptionAPI
from audio_transcribe.utils.api.chunking import ChunkingMixin
from audio_transcribe.transcribe_helpers.audio_processing import convert_to_flac, get_api_file_size_limit

class GroqAPI(TranscriptionAPI, ChunkingMixin):
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
            import groq
            self.groq = groq
            
            # Log masked API key for debugging
            if self.api_key:
                masked_key = self.mask_api_key(self.api_key)
                logger.debug(f"Initializing Groq client with API key: {masked_key}")
                
            self.client = groq.Groq(api_key=self.api_key) if self.api_key else None
        except ImportError:
            logger.error("Groq package not found. Please install it: uv add groq")
            self.client = None
            
    def list_models(self) -> List[str]:
        """
        List available models for Groq API.
        
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
            logger.error(f"Failed to list Groq models: {e}")
            return []

    def check_api_key(self) -> bool:
        """Check if Groq API key is valid."""
        if not self.api_key:
            logger.error("No Groq API key provided")
            return False
            
        if not self.client:
            logger.error("Groq client not initialized")
            return False
            
        try:
            # Use list_models to validate API key
            models = self.list_models()
            if models:
                logger.debug(f"Groq API key valid. Available models: {len(models)}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to validate Groq API key: {str(e)}")
            return False
            
    def transcribe_chunk(self, chunk_path: Path, chunk_index: int, start_time: float, **kwargs) -> Tuple[Dict[str, Any], float]:
        """
        Transcribe a single audio chunk using Groq's audio transcriptions API.
        
        Args:
            chunk_path: Path to the audio chunk file
            chunk_index: Index of the chunk
            start_time: Start time of this chunk in seconds
            **kwargs: Additional arguments (model, language, etc.)
            
        Returns:
            Tuple of (chunk_result, chunk_start_seconds)
        """
        if not self.client:
            raise ValueError("Groq client not initialized")
            
        # Convert Path to string if needed
        audio_chunk_path = str(chunk_path)
        
        # Validate chunk size (Groq limit is 25MB)
        chunk_size_mb = os.path.getsize(chunk_path) / (1024 * 1024)
        limit_mb = get_api_file_size_limit("groq")
        if chunk_size_mb > limit_mb:
            logger.warning(f"Chunk {chunk_index} size ({chunk_size_mb:.2f}MB) exceeds {limit_mb}MB limit. "
                          f"This may cause API errors.")
        
        model = kwargs.get("model", "whisper-large-v3")
        language = kwargs.get("language")
            
        # Ensure model is never None
        if model is None:
            model = "whisper-large-v3"
            
        logger.info(f"Transcribing chunk {chunk_index} starting at {start_time:.1f}s with Groq (model: {model}, size: {chunk_size_mb:.2f}MB)")
        
        # Open the file in binary mode
        with open(audio_chunk_path, "rb") as audio_file:
            try:
                # Use the audio.transcriptions.create endpoint
                start_time_perf = time.time()
                logger.debug(f"Calling Groq API with model={model}, language={language if language else 'None'}")
                
                # Prepare arguments
                api_kwargs = {
                    "file": ("chunk.flac", audio_file, "audio/flac"),
                    "model": model,
                    "response_format": "verbose_json",
                    "temperature": 0,  # For best transcription quality
                    "timestamp_granularities": ["word", "segment"]  # Request both word and segment-level timestamps
                }
                
                if language:
                    api_kwargs["language"] = language
                
                result = self.client.audio.transcriptions.create(**api_kwargs)
                
                transcription_time = time.time() - start_time_perf
                logger.info(f"Chunk processed in {transcription_time:.2f}s")
                
                # Extract data from the result
                if hasattr(result, 'model_dump'):
                    # Handle Pydantic model response (newer Groq SDK)
                    raw_data = result.model_dump()
                else:
                    # Handle dict-like response
                    raw_data = dict(result)
                
                # Save the raw response for this chunk
                chunk_file_path = Path(audio_chunk_path)
                raw_chunk_json_path = chunk_file_path.with_name(f"{chunk_file_path.stem}_{self.api_name}.json")
                try:
                    with open(raw_chunk_json_path, 'w', encoding='utf-8') as f_raw_chunk:
                        json.dump(raw_data, f_raw_chunk, indent=2, ensure_ascii=False)
                    logger.info(f"Saved raw Groq chunk response to {raw_chunk_json_path}")
                except Exception as e_raw_chunk_save:
                    logger.error(f"Failed to save raw Groq chunk response: {e_raw_chunk_save}")

                return raw_data, start_time
                
            except Exception as e:
                logger.error(f"Error transcribing chunk with Groq: {str(e)}")
                raise ValueError(f"Groq transcription failed: {str(e)}")
            
    def merge_chunk_results(self, results: List[Tuple[Dict[str, Any], float]]) -> Dict[str, Any]:
        """
        Merge transcription chunks, handling word-level timestamps.
        
        Args:
            results: List of (result, start_time_seconds) tuples
            
        Returns:
            Merged transcription result dict
        """
        logger.info("Merging transcription chunks")
        
        # Initialize merged result
        merged_segments = []
        all_words = []
        
        for i, (chunk, chunk_start_sec) in enumerate(results):
            # Extract segments and words if available
            segments = chunk.get('segments', [])
            words = chunk.get('words', [])
            
            # Adjust segment timestamps (chunk timestamps are relative, add absolute start time)
            for segment in segments:
                adjusted_segment = segment.copy()
                adjusted_segment['start'] = segment.get('start', 0) + chunk_start_sec
                adjusted_segment['end'] = segment.get('end', 0) + chunk_start_sec
                merged_segments.append(adjusted_segment)
            
            # Adjust word timestamps
            adjusted_words = []
            for word in words:
                adjusted_word = word.copy()
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

    def transcribe(self, audio_path: Union[str, Path], **kwargs) -> TranscriptionResult:
        """
        Transcribe audio file using Groq with proper chunking for long files.
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
        flac_path = convert_to_flac(audio_path)
        if not flac_path or not os.path.exists(flac_path):
            logger.error(f"Failed to convert audio to FLAC: {audio_path}")
            return None
            
        try:
            # Step 2: Check if we need to chunk the audio
            from pydub import AudioSegment
            audio = AudioSegment.from_file(flac_path)
            duration_seconds = len(audio) / 1000.0  # Duration in seconds
            logger.info(f"Audio duration: {duration_seconds:.2f} seconds")
            
            # If audio is short enough and under size limit, transcribe in a single request
            limit_mb = get_api_file_size_limit("groq")
            file_size_mb = os.path.getsize(flac_path) / (1024 * 1024)
            
            if duration_seconds <= chunk_length and file_size_mb <= limit_mb:
                logger.info("Audio is short enough for single transcription request")
                result_dict, _ = self.transcribe_chunk(Path(flac_path), 0, 0.0, **kwargs)
            else:
                # Log chunking plan
                logger.info(f"Using chunking: length={chunk_length}s, overlap={overlap}s")
                # Use ChunkingMixin
                result_dict = self.transcribe_with_chunking(flac_path, chunk_length, overlap, **kwargs)
                
            if not result_dict:
                return None

            # Add API name to the response
            result_dict["api_name"] = self.api_name
            
            # Save raw result for debugging and reference
            self.save_result(result_dict, audio_path)
            
            # Import here to avoid circular imports
            result_obj = parse_groq_format(result_dict)
            
            return result_obj
            
        except Exception as e:
            logger.error(f"Error transcribing with Groq: {str(e)}")
            raise ValueError(f"Groq transcription failed: {str(e)}")
        finally:
            # Clean up temporary FLAC file
            keep_flac = kwargs.get("keep_flac", False)
            if not keep_flac and flac_path and os.path.exists(flac_path) and os.path.basename(flac_path) != os.path.basename(audio_path):
                try:
                    os.unlink(flac_path)
                except:
                    pass
