"""
Transcription API package.
"""
from typing import Optional

from audio_transcribe.utils.api.base import TranscriptionAPI
from audio_transcribe.utils.api.assemblyai import AssemblyAIAPI
from audio_transcribe.utils.api.elevenlabs import ElevenLabsAPI
from audio_transcribe.utils.api.groq import GroqAPI
from audio_transcribe.utils.api.openai import OpenAIAPI

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

__all__ = [
    "TranscriptionAPI",
    "AssemblyAIAPI",
    "ElevenLabsAPI",
    "GroqAPI",
    "OpenAIAPI",
    "get_api_instance"
]
