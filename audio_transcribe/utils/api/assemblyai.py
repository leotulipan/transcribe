"""
AssemblyAI API implementation.
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from loguru import logger

from audio_transcribe.utils.parsers import TranscriptionResult, parse_assemblyai_format
from audio_transcribe.utils.api.base import TranscriptionAPI
from audio_transcribe.utils.adapters import ParameterAdapter
from audio_transcribe.transcribe_helpers.audio_processing import extract_audio_from_mp4, check_file_size, get_api_file_size_limit

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
            
            # Log masked API key for debugging
            if self.api_key:
                masked_key = self.mask_api_key(self.api_key)
                logger.debug(f"Initializing AssemblyAI with API key: {masked_key}")
                aai.settings.api_key = self.api_key
                
        except ImportError:
            logger.error("AssemblyAI package not found. Please install it: uv add assemblyai")
            self.aai = None
            
    def list_models(self) -> List[str]:
        """
        List available models for AssemblyAI API.
        
        Returns:
            List of model IDs available for use
        """
        # AssemblyAI has static model names
        return ["best", "nano"]

    def check_api_key(self) -> bool:
        """Check if AssemblyAI API key is valid."""
        if not self.api_key:
            logger.error("No AssemblyAI API key provided")
            return False
            
        if not self.aai:
            logger.error("AssemblyAI package not loaded")
            return False
            
        # The AssemblyAI SDK doesn't have a lightweight 'validate' method without transcribing.
        # We assume if the key is set, it's potentially valid. 
        # Real validation happens during transcription.
        return True
            
    def transcribe(self, audio_path: Union[str, Path], **kwargs) -> TranscriptionResult:
        """
        Transcribe audio file using AssemblyAI.
        
        Args:
            audio_path: Path to the audio file
            **kwargs: Additional AssemblyAI-specific parameters:
                - language: Language code
                - speaker_labels: Enable speaker diarization
                - dual_channel: Enable dual channel transcription
                - model: Model to use (best, nano, etc.)
                
        Returns:
            Standardized TranscriptionResult object
        """
        if not self.aai:
            raise ValueError("AssemblyAI package not loaded")
            
        # Convert Path to string if needed
        if isinstance(audio_path, Path):
            audio_path = str(audio_path)
            
        # Check if input is MP4/Video and extract audio if needed
        temp_audio_path = None
        processing_path = audio_path
        
        # Check if input is MP4/Video and extract audio if needed
        # NOTE: Audio optimization and extraction is now handled in cli.py before calling transcribe
        # We just log the file being used
        logger.info(f"Transcribing file: {audio_path}")
        processing_path = audio_path
            
        # Normalize parameters using ParameterAdapter
        adapted_params = ParameterAdapter.adapt_for_api("assemblyai", kwargs)
        
        # Prepare transcription config
        model = adapted_params.get("model", "best")
        
        config_params = {
            "speaker_labels": adapted_params.get("speaker_labels", True),
            "dual_channel": adapted_params.get("dual_channel", False),
            "speech_model": model if model in ["best", "nano"] else "best" 
        }
        
        # Handle language parameter
        if "language_code" in adapted_params:
            config_params["language_code"] = adapted_params["language_code"]
        else:
            config_params["language_detection"] = True
            
        config = self.aai.TranscriptionConfig(**config_params)
        
        logger.info(f"Transcribing {processing_path} with AssemblyAI (model: {model})")
        
        # Submit and wait for completion
        try:
            transcriber = self.aai.Transcriber()
            transcript = transcriber.transcribe(processing_path, config=config)
            
            if transcript.status == self.aai.TranscriptStatus.error:
                raise ValueError(f"AssemblyAI transcription failed: {transcript.error}")
                
            logger.info(f"Transcription completed: {transcript.id}")
            
            # Convert to standardized format
            result_dict = transcript.json_response
            result_dict["api_name"] = self.api_name
            
            # Save raw JSON response (using original audio path for naming)
            self.save_result(result_dict, audio_path)
            
            # Parse result
            result = parse_assemblyai_format(result_dict)
            
            return result
            
        except Exception as e:
            logger.error(f"AssemblyAI transcription failed: {str(e)}")
            raise
        finally:
            # Clean up temporary file if created
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.unlink(temp_audio_path)
                    logger.info(f"Deleted temporary audio file: {temp_audio_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {temp_audio_path}: {e}")
