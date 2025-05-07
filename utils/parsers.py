"""
Standardized parsers for different transcription API formats.

This module provides a unified way to parse JSON responses from different
transcription APIs (AssemblyAI, ElevenLabs, Groq, OpenAI) into a consistent format.
"""
from typing import Dict, Any, List, Optional, Union
import json
from loguru import logger
from pathlib import Path


class TranscriptionResult:
    """Unified data model for transcription results from any API."""
    
    def __init__(self, text: str = "", words: List[Dict[str, Any]] = None, 
                 language: str = "", api_name: str = "", 
                 speaker_count: int = 0, speakers: List[Dict[str, Any]] = None):
        self.text = text
        self.words = words or []
        self.language = language
        self.api_name = api_name
        self.speaker_count = speaker_count
        self.speakers = speakers or []
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the transcription result to a dictionary."""
        return {
            "text": self.text,
            "words": self.words,
            "language": self.language,
            "api_name": self.api_name,
            "speaker_count": self.speaker_count,
            "speakers": self.speakers
        }
        
    def to_json(self, indent: int = 2) -> str:
        """Convert the transcription result to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def save(self, file_path: Union[str, Path]) -> None:
        """Save the transcription result to a JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
            
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TranscriptionResult':
        """Create a TranscriptionResult from a dictionary."""
        return cls(
            text=data.get("text", ""),
            words=data.get("words", []),
            language=data.get("language", ""),
            api_name=data.get("api_name", ""),
            speaker_count=data.get("speaker_count", 0),
            speakers=data.get("speakers", [])
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'TranscriptionResult':
        """Create a TranscriptionResult from a JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> 'TranscriptionResult':
        """Load a TranscriptionResult from a JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return cls.from_dict(json.load(f))


def parse_assemblyai_format(data: Dict[str, Any]) -> TranscriptionResult:
    """
    Parse AssemblyAI transcription data into standardized format.
    
    Args:
        data: AssemblyAI response JSON data
        
    Returns:
        Standardized TranscriptionResult object
    """
    logger.debug("Parsing AssemblyAI format")
    
    # Extract text and basic metadata
    text = data.get("text", "")
    language = data.get("language_code", "")
    
    # Process words data
    words = []
    if "words" in data and data["words"]:
        for word_data in data["words"]:
            word = {
                "text": word_data.get("text", ""),
                "start": word_data.get("start") / 1000.0 if word_data.get("start") is not None else 0,  # Convert ms to seconds
                "end": word_data.get("end") / 1000.0 if word_data.get("end") is not None else 0,        # Convert ms to seconds
                "type": "word",
                "speaker_id": word_data.get("speaker", "Unknown")
            }
            words.append(word)
    
    # Process speaker data
    speakers = []
    if "speaker_labels" in data and data["speaker_labels"]:
        speaker_data = data["speaker_labels"].get("speakers", [])
        for i, speaker in enumerate(speaker_data):
            speakers.append({
                "id": speaker,
                "name": f"Speaker {i+1}"
            })
    
    return TranscriptionResult(
        text=text,
        words=words,
        language=language,
        api_name="assemblyai",
        speaker_count=len(speakers),
        speakers=speakers
    )


def parse_elevenlabs_format(data: Dict[str, Any]) -> TranscriptionResult:
    """
    Parse ElevenLabs transcription data into standardized format.
    
    Args:
        data: ElevenLabs response JSON data
        
    Returns:
        Standardized TranscriptionResult object
    """
    logger.debug("Parsing ElevenLabs format")
    
    # Extract text and basic metadata
    text = data.get("text", "")
    language = data.get("language", "")
    
    # Process words data - ElevenLabs already has times in seconds
    words = data.get("words", [])
    
    # ElevenLabs doesn't provide speaker data
    speakers = []
    
    return TranscriptionResult(
        text=text,
        words=words,
        language=language,
        api_name="elevenlabs",
        speaker_count=0,
        speakers=speakers
    )


def parse_groq_format(data: Dict[str, Any]) -> TranscriptionResult:
    """
    Parse Groq transcription data into standardized format.
    
    Args:
        data: Groq response JSON data
        
    Returns:
        Standardized TranscriptionResult object
    """
    logger.debug("Parsing Groq format")
    
    # Extract text and basic metadata
    text = data.get("text", "")
    language = data.get("language", "")
    
    # Process words data - Groq times are in seconds
    words = data.get("words", [])
    
    # Groq currently doesn't provide speaker data
    speakers = []
    
    return TranscriptionResult(
        text=text,
        words=words,
        language=language,
        api_name="groq",
        speaker_count=0,
        speakers=speakers
    )


def parse_openai_format(data: Dict[str, Any]) -> TranscriptionResult:
    """
    Parse OpenAI Whisper transcription data into standardized format.
    
    Args:
        data: OpenAI response JSON data
        
    Returns:
        Standardized TranscriptionResult object
    """
    logger.debug("Parsing OpenAI format")
    
    # Extract text and basic metadata
    text = data.get("text", "")
    language = data.get("language", "")
    
    # Process words data if available (OpenAI Whisper might not provide word-level timestamps by default)
    words = []
    if "words" in data and data["words"]:
        for word_data in data["words"]:
            word = {
                "text": word_data.get("text", ""),
                "start": word_data.get("start", 0),  # OpenAI uses seconds
                "end": word_data.get("end", 0),      # OpenAI uses seconds
                "type": "word"
            }
            words.append(word)
    
    return TranscriptionResult(
        text=text,
        words=words,
        language=language,
        api_name="openai",
        speaker_count=0,
        speakers=[]
    )


def detect_and_parse_json(data: Dict[str, Any]) -> TranscriptionResult:
    """
    Auto-detect the JSON format and parse it into the standardized format.
    
    Args:
        data: JSON data from any supported API
        
    Returns:
        Standardized TranscriptionResult object
    """
    # Check if the api_name is already in the data
    if "api_name" in data:
        api_name = data["api_name"]
        logger.info(f"Found api_name in data: {api_name}")
        
        if api_name == "assemblyai":
            return parse_assemblyai_format(data)
        elif api_name == "elevenlabs":
            return parse_elevenlabs_format(data)
        elif api_name == "groq":
            return parse_groq_format(data)
        elif api_name == "openai":
            return parse_openai_format(data)
    
    # Try to detect format based on data structure
    if "audio_url" in data or "status" in data and "words" in data:
        return parse_assemblyai_format(data)
    elif "words" in data and any(word.get("type") == "spacing" for word in data.get("words", [])):
        return parse_elevenlabs_format(data)
    elif "model" in data and data.get("model", "").startswith("whisper-"):
        return parse_openai_format(data)
    elif "text" in data and "words" in data:
        # This is a bit generic, but we'll assume Groq for now
        return parse_groq_format(data)
    
    # If we can't determine the format, create a basic result with the original data
    logger.warning("Could not determine JSON format, creating basic result")
    return TranscriptionResult(
        text=data.get("text", ""),
        words=data.get("words", []),
        api_name="unknown"
    )


def load_and_parse_json(file_path: Union[str, Path]) -> TranscriptionResult:
    """
    Load a JSON file and parse it into the standardized format.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Standardized TranscriptionResult object
    """
    logger.info(f"Loading and parsing JSON file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    return detect_and_parse_json(data) 