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
from audio_transcribe.transcribe_helpers.audio_processing import convert_to_flac, get_api_file_size_limit
from audio_transcribe.transcribe_helpers.chunking import split_audio

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
            
        # Ensure model is never None
        if model is None:
            model = "whisper-large-v3"
            logger.info(f"No model specified for chunk, using default: {model}")
            
        logger.info(f"Transcribing chunk starting at {chunk_start_ms}ms with Groq (model: {model})")
        
        # Open the file in binary mode
        with open(audio_chunk_path, "rb") as audio_file:
            try:
                # Use the audio.transcriptions.create endpoint
                # Pass the file directly as a tuple (filename, fileobj, content_type)
                start_time = time.time()
                logger.debug(f"Calling Groq API with model={model}, language={language if language else 'None'}")
                
                # Prepare arguments
                kwargs = {
                    "file": ("chunk.flac", audio_file, "audio/flac"),
                    "model": model,
                    "response_format": "verbose_json",
                    "temperature": 0,  # For best transcription quality
                    "timestamp_granularities": ["word", "segment"]  # Request both word and segment-level timestamps
                }
                
                if language:
                    kwargs["language"] = language
                
                result = self.client.audio.transcriptions.create(**kwargs)
                
                transcription_time = time.time() - start_time
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

                return raw_data, chunk_start_ms
                
            except Exception as e:
                logger.error(f"Error transcribing chunk with Groq: {str(e)}")
                raise ValueError(f"Groq transcription failed: {str(e)}")
            
    def transcribe(self, audio_path: Union[str, Path], **kwargs) -> TranscriptionResult:
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
            Standardized TranscriptionResult object
        """
        if not self.client:
            raise ValueError("Groq client not initialized")
            
        # Extract parameters
        language = kwargs.get("language")
        model = kwargs.get("model", "whisper-large-v3")
        
        # Always ensure we have a valid model - never pass None
        if model is None:
            model = "whisper-large-v3"
            logger.info(f"No model specified, using default: {model}")
            
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
            duration = len(audio)  # Duration in milliseconds
            logger.info(f"Audio duration: {duration/1000:.2f} seconds")
            
            # If audio is short enough and under size limit, transcribe in a single request
            limit_mb = get_api_file_size_limit("groq")
            file_size_mb = os.path.getsize(flac_path) / (1024 * 1024)
            
            if duration <= chunk_length * 1000 and file_size_mb <= limit_mb:
                logger.info("Audio is short enough for single transcription request")
                result_dict, _ = self.transcribe_chunk(flac_path, 0, model, language)
                
            else:
                # Step 3: Split audio into chunks and transcribe each
                logger.info(f"Audio is {duration/1000:.2f}s long, splitting into chunks of {chunk_length}s with {overlap}s overlap")
                
                chunk_ms = chunk_length * 1000
                overlap_ms = overlap * 1000
                total_chunks = (duration // (chunk_ms - overlap_ms)) + 1
                
                results = []
                
                for i in range(int(total_chunks)):
                    start = i * (chunk_ms - overlap_ms)
                    end = min(start + chunk_ms, duration)
                    
                    # Stop if start is beyond duration
                    if start >= duration:
                        break
                        
                    logger.info(f"Processing chunk {i+1}/{int(total_chunks) + 1} ({start/1000:.1f}s - {end/1000:.1f}s)")
                    
                    # Extract chunk
                    chunk = audio[start:end]
                    
                    # Save chunk to temporary file
                    with tempfile.NamedTemporaryFile(suffix='.flac', delete=False) as temp_file:
                        chunk_path = temp_file.name
                        
                    chunk.export(chunk_path, format='flac')
                    
                    # Transcribe chunk
                    try:
                        result, _ = self.transcribe_chunk(chunk_path, start, model, language)
                        results.append((result, start))
                    except Exception as e:
                        logger.error(f"Error transcribing chunk {i+1}: {str(e)}")
                        raise
                    finally:
                        # Clean up temp file
                        if os.path.exists(chunk_path):
                            os.unlink(chunk_path)
                
                # Step 4: Merge chunks
                result_dict = self._merge_transcripts(results, overlap)
                
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
