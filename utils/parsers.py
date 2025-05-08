"""
Standardized parsers for different transcription API formats.

This module provides a unified way to parse JSON responses from different
transcription APIs (AssemblyAI, ElevenLabs, Groq, OpenAI) into a consistent format.
"""
from typing import Dict, Any, List, Optional, Union, Tuple
import json
from loguru import logger
from pathlib import Path

# Custom JSON encoder to format floats without scientific notation
class CustomJSONEncoder(json.JSONEncoder):
    def encode(self, obj):
        if isinstance(obj, float):
            # Format float to avoid scientific notation for very small numbers
            return format(obj, 'f')
        return super().encode(obj)
        
    def iterencode(self, obj, _one_shot=False):
        if isinstance(obj, dict):
            # Format floats in dictionaries to avoid scientific notation
            formatted_dict = {}
            for k, v in obj.items():
                if isinstance(v, float):
                    formatted_dict[k] = format(v, 'f')
                else:
                    formatted_dict[k] = v
            return super().iterencode(formatted_dict, _one_shot)
        elif isinstance(obj, list):
            # Format floats in lists to avoid scientific notation
            formatted_list = []
            for item in obj:
                if isinstance(item, dict):
                    formatted_item = {}
                    for k, v in item.items():
                        if isinstance(v, float):
                            formatted_item[k] = format(v, 'f')
                        else:
                            formatted_item[k] = v
                    formatted_list.append(formatted_item)
                else:
                    formatted_list.append(item)
            return super().iterencode(formatted_list, _one_shot)
        elif isinstance(obj, float):
            # Format float to avoid scientific notation
            return format(obj, 'f')
        return super().iterencode(obj, _one_shot)


class TranscriptionResult:
    """
    Standardized representation of a transcription result.
    
    This class provides a common interface for working with transcription results
    from different APIs, ensuring consistent access to the data regardless of the
    original format.
    """
    
    def __init__(self, text: str = "", confidence: float = 0.0, language: str = "en",
                words: List[Dict[str, Any]] = None, speakers: List[Dict[str, Any]] = None,
                api_name: str = "unknown"):
        """
        Initialize a TranscriptionResult.
        
        Args:
            text: Full transcript text
            confidence: Overall confidence score (0.0-1.0)
            language: Language code (ISO-639-1 or ISO-639-3)
            words: List of word dictionaries with timing info
            speakers: List of speaker dictionaries with identification info
            api_name: Name of the API used for transcription
        """
        self.text = text
        self.confidence = confidence
        self.language = language
        self.words = words or []
        self.speakers = speakers or []
        self.api_name = api_name
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the transcription result to a dictionary."""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "language": self.language,
            "api_name": self.api_name,
            "words": self.words,
            "speakers": self.speakers
        }
    
    def to_words_json(self, indent: int = 2) -> str:
        """Convert just the words data to a JSON string."""
        # Process words to ensure proper float formatting
        processed_words = []
        for word in self.words:
            processed_word = {}
            for k, v in word.items():
                if isinstance(v, float):
                    # Format float values to 4 decimal places max without scientific notation
                    processed_word[k] = format(v, '.4f')
                else:
                    processed_word[k] = v
            processed_words.append(processed_word)
            
        return json.dumps({"words": processed_words}, indent=indent, ensure_ascii=False)
        
    def to_json(self, indent: int = 2) -> str:
        """Convert the transcription result to a JSON string."""
        # Create a processed copy with formatted float values
        processed_dict = self.to_dict()
        
        # Process words list
        processed_words = []
        for word in processed_dict['words']:
            processed_word = {}
            for k, v in word.items():
                if isinstance(v, float):
                    # Format float values to 4 decimal places max without scientific notation
                    processed_word[k] = format(v, '.4f')
                else:
                    processed_word[k] = v
            processed_words.append(processed_word)
            
        processed_dict['words'] = processed_words
        
        return json.dumps(processed_dict, indent=indent, ensure_ascii=False)
    
    def save(self, file_path: Union[str, Path]) -> None:
        """
        Save transcription result to a JSON file.
        
        Args:
            file_path: Path to save the JSON file
        """
        data = {
            "text": self.text,
            "confidence": self.confidence,
            "language": self.language,
            "words": self.words,
            "speakers": self.speakers,
            "api_name": self.api_name
        }
        
        # Use a custom JSON serialization approach to prevent scientific notation for small numbers
        class CustomJSONEncoder(json.JSONEncoder):
            def iterencode(self, obj, **kwargs):
                # Special handling for TranscriptionResult's words list
                if obj == data and "words" in obj and obj["words"]:
                    # Start with open brace
                    yield "{\n"
                    
                    # Handle non-words fields first
                    non_words_items = [(k, v) for k, v in obj.items() if k != "words"]
                    for i, (key, value) in enumerate(non_words_items):
                        yield f'  "{key}": '
                        yield from super().iterencode(value)
                        if i < len(non_words_items) - 1 or "words" in obj:
                            yield ",\n"
                        else:
                            yield "\n"
                    
                    # Now handle words with custom formatting for start/end timestamps
                    if "words" in obj:
                        words = obj["words"]
                        yield '  "words": [\n'
                        
                        for i, word in enumerate(words):
                            word_copy = word.copy()
                            
                            # Ensure speaker_id is a simple string
                            if "speaker_id" in word_copy and (word_copy["speaker_id"] is None or word_copy["speaker_id"] == "Unknown"):
                                word_copy["speaker_id"] = ""
                            
                            # Ensure start/end are proper integers if they're milliseconds or fixed precision floats
                            # This prevents scientific notation in small decimals
                            yield "    {\n"
                            
                            word_items = list(word_copy.items())
                            for j, (k, v) in enumerate(word_items):
                                if k in ["start", "end"] and isinstance(v, (int, float)):
                                    # Format floating point numbers with fixed precision
                                    yield f'      "{k}": {v}'
                                else:
                                    # Use standard JSON encoding for other fields
                                    yield f'      "{k}": '
                                    yield from super().iterencode(v)
                                
                                if j < len(word_items) - 1:
                                    yield ",\n"
                                else:
                                    yield "\n"
                            
                            if i < len(words) - 1:
                                yield "    },\n"
                            else:
                                yield "    }\n"
                        
                        yield "  ]\n"
                    
                    # Close the object
                    yield "}"
                else:
                    # For other objects, use standard JSON encoding
                    yield from super().iterencode(obj)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                # Use our custom encoder
                logger.debug(f"Saving transcription result with custom JSON encoder to prevent scientific notation")
                for chunk in CustomJSONEncoder(ensure_ascii=False, indent=2).iterencode(data):
                    f.write(chunk)
        except Exception as e:
            logger.error(f"Failed to save transcription result: {str(e)}")
            
    def save_words_only(self, file_path: Union[str, Path]) -> None:
        """
        Save only the words array to a JSON file.
        
        Args:
            file_path: Path to save the JSON file with only words array
        """
        if not self.words:
            logger.warning("Words array is empty, saving empty words array")
            
        try:
            # Create a simplified encoder for just the words array
            class WordsOnlyJSONEncoder(json.JSONEncoder):
                def iterencode(self, obj, **kwargs):
                    if isinstance(obj, list) and obj == self.words:
                        # Start the words array
                        yield "[\n"
                        
                        for i, word in enumerate(obj):
                            word_copy = word.copy()
                            
                            # Ensure speaker_id is a simple string
                            if "speaker_id" in word_copy and (word_copy["speaker_id"] is None or word_copy["speaker_id"] == "Unknown"):
                                word_copy["speaker_id"] = ""
                            
                            yield "  {\n"
                            
                            word_items = list(word_copy.items())
                            for j, (k, v) in enumerate(word_items):
                                if k in ["start", "end"] and isinstance(v, (int, float)):
                                    # Format floating point numbers with fixed precision to avoid scientific notation
                                    yield f'    "{k}": {v}'
                                else:
                                    # Use standard JSON encoding for other fields
                                    yield f'    "{k}": '
                                    yield from super().iterencode(v)
                                
                                if j < len(word_items) - 1:
                                    yield ",\n"
                                else:
                                    yield "\n"
                            
                            if i < len(obj) - 1:
                                yield "  },\n"
                            else:
                                yield "  }\n"
                        
                        # Close the array
                        yield "]"
                    else:
                        # For non-words lists, use standard encoding
                        yield from super().iterencode(obj)
            
            # Create custom encoder with reference to words
            encoder = WordsOnlyJSONEncoder(ensure_ascii=False, indent=2)
            encoder.words = self.words
            
            with open(file_path, 'w', encoding='utf-8') as f:
                logger.debug(f"Saving words-only JSON to {file_path}")
                for chunk in encoder.iterencode(self.words):
                    f.write(chunk)
                    
        except Exception as e:
            logger.error(f"Failed to save words-only JSON: {str(e)}")
            
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TranscriptionResult':
        """Create a TranscriptionResult from a dictionary."""
        return cls(
            text=data.get("text", ""),
            confidence=data.get("confidence", 0.0),
            language=data.get("language", "en"),
            words=data.get("words", []),
            speakers=data.get("speakers", []),
            api_name=data.get("api_name", "unknown")
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
    Parse AssemblyAI format to create a TranscriptionResult.
    
    Args:
        data: Dictionary containing AssemblyAI API response data
        
    Returns:
        TranscriptionResult object
    """
    logger.debug("Parsing AssemblyAI format (basic)")
    
    # Extract basic info
    text = data.get('text', '')
    confidence = data.get('confidence', 0.0)
    language = data.get('language', data.get('language_code', 'en'))
    
    # Get words or utterances
    raw_words = data.get('words', [])
    
    # AssemblyAI doesn't explicitly mark spaces, so we'll need to identify them
    # during standardization in a separate step
    
    # Convert words to our standard format
    words = []
    scientific_notation_count = 0
    
    for word in raw_words:
        word_text = word.get('text', '')
        start = word.get('start', 0)
        end = word.get('end', 0)
        confidence = word.get('confidence', 0.0)
        speaker = word.get('speaker', '')
        
        # Fix scientific notation in timestamps (e.g. 2.4e-07)
        if isinstance(start, (int, float)):
            # Check for very small values (likely scientific notation)
            start_str = str(start)
            if 'e' in start_str.lower() or abs(start) < 0.0001 and start != 0:
                scientific_notation_count += 1
                logger.warning(f"Fixed scientific notation or very small timestamp: {start} → 0")
                start = 0
                
        if isinstance(end, (int, float)):
            # Check for very small values (likely scientific notation)
            end_str = str(end)
            if 'e' in end_str.lower() or abs(end) < 0.0001 and end != 0:
                scientific_notation_count += 1
                logger.warning(f"Fixed scientific notation or very small timestamp: {end} → 0")
                end = 0
        
        # Identify spaces (should be marked as spacing type, not word) 
        if word_text.strip() == '' or word_text.isspace():
            word_type = 'spacing'
        else:
            word_type = 'word'
            
        # Create standardized word entry
        word_entry = {
            'text': word_text,
            'start': start,
            'end': end,
            'confidence': confidence,
            'type': word_type
        }
        
        # Add speaker if available
        if speaker:
            word_entry['speaker_id'] = speaker
            
        words.append(word_entry)
    
    if scientific_notation_count > 0:
        logger.info(f"Fixed {scientific_notation_count} scientific notation or very small timestamps")
    
    # Create and return result
    return TranscriptionResult(
        text=text,
        confidence=confidence,
        language=language,
        words=words,
        speakers=[],
        api_name='assemblyai'
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
    
    words = []
    words_data = None
    # Support both top-level 'words' and 'segments' (web app export)
    if "words" in data and data["words"]:
        words_data = data["words"]
    elif "segments" in data and isinstance(data["segments"], list):
        # Flatten all segment['words'] into a single list
        words_data = []
        for segment in data["segments"]:
            if isinstance(segment, dict) and "words" in segment and isinstance(segment["words"], list):
                words_data.extend(segment["words"])
    
    if words_data:
        for word_data in words_data:
            if isinstance(word_data, dict) and "text" in word_data:
                words.append({
                    "text": word_data.get("text", ""),
                    "start": word_data.get("start", 0),
                    "end": word_data.get("end", 0),
                    "type": word_data.get("type", "word")
                })
    else:
        # If no words data but text is available, generate simple word objects
        if text:
            logger.info("Generating words from text content since words array is empty")
            words = generate_words_from_text(text)
        else:
            logger.warning("No words data found in ElevenLabs format and no text available")
    
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
    words = []
    if "words" in data and data["words"]:
        # Ensure we're getting a proper list of word objects
        for word_data in data["words"]:
            # Make sure each word has the necessary fields
            if isinstance(word_data, dict) and "text" in word_data:
                words.append({
                    "text": word_data.get("text", ""),
                    "start": word_data.get("start", 0),
                    "end": word_data.get("end", 0),
                    "type": word_data.get("type", "word")
                })
    else:
        # If no words data but text is available, generate simple word objects
        if text:
            logger.info("Generating words from text content since words array is empty")
            words = generate_words_from_text(text)
        else:
            logger.warning("No words data found in Groq format and no text available")
    
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
            if isinstance(word_data, dict) and "text" in word_data:
                word = {
                    "text": word_data.get("text", ""),
                    "start": word_data.get("start", 0),  # OpenAI uses seconds
                    "end": word_data.get("end", 0),      # OpenAI uses seconds
                    "type": "word"
                }
                words.append(word)
    else:
        # If no words data but text is available, generate simple word objects
        if text:
            logger.info("Generating words from text content since words array is empty")
            words = generate_words_from_text(text)
        else:
            logger.warning("No words data found in OpenAI format and no text available")
    
    return TranscriptionResult(
        text=text,
        words=words,
        language=language,
        api_name="openai",
        speaker_count=0,
        speakers=[]
    )


def generate_words_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Generate word-level objects from plain text.
    
    Args:
        text: Plain text to convert to word objects
        
    Returns:
        List of word objects with approximate timing
    """
    if not text:
        return []
    
    # Split text into words
    raw_words = text.split()
    if not raw_words:
        return []
    
    # Generate approximate timing (1 second per word as a simple estimate)
    words = []
    
    # Start with words
    current_time = 0.0
    for i, word_text in enumerate(raw_words):
        # Estimate word duration based on length (roughly 0.15s per character)
        duration = max(0.3, len(word_text) * 0.15)
        
        # Create word object
        word = {
            "text": word_text,
            "start": current_time,
            "end": current_time + duration,
            "type": "word"
        }
        words.append(word)
        
        # Add spacing after word (except the last word)
        if i < len(raw_words) - 1:
            space = {
                "text": " ",
                "start": current_time + duration,
                "end": current_time + duration + 0.1,
                "type": "spacing"
            }
            words.append(space)
            current_time += duration + 0.1
        else:
            current_time += duration
    
    return words


def detect_and_parse_json(data: Dict[str, Any]) -> Tuple[str, Union[List[Dict[str, Any]], TranscriptionResult]]:
    """
    Auto-detect the JSON format and parse it into a TranscriptionResult or basic word list.
    
    Args:
        data: JSON data from any supported API
        
    Returns:
        Tuple containing detected api_name (str) and either a TranscriptionResult object 
        or a List of word dicts.
    """
    detected_api = "unknown"
    result = None
    
    # Check if the api_name is already in the data
    if "api_name" in data:
        api_name = data["api_name"]
        logger.info(f"Found api_name in data: {api_name}")
        detected_api = api_name
        if api_name == "assemblyai":
            result = parse_assemblyai_format(data)
        # ... Add elif blocks for other APIs calling their respective parsers ...
        # elif api_name == "elevenlabs":
        #     result = parse_elevenlabs_format(data) 
        # elif api_name == "groq":
        #     result = parse_groq_format(data)
        # elif api_name == "openai":
        #     result = parse_openai_format(data)
        else:
             logger.warning(f"API name '{api_name}' found but no specific parser implemented.")
             # Fallback to generic detection or text generation if possible
             if "words" in data:
                 words = data.get("words", []) # Assume basic format or handle later
                 result = TranscriptionResult(
                     text=data.get("text", ""),
                     words=words,
                     language=data.get("language", ""),
                     api_name=api_name
                 )
             elif "text" in data:
                 words = generate_words_from_text(data.get("text", ""))
                 result = TranscriptionResult(
                     text=data.get("text", ""),
                     words=words,
                     language=data.get("language", ""),
                     api_name=api_name
                 )
             
    else:
        # Try to detect format based on data structure
        if "audio_url" in data or "status" in data and "words" in data:
            detected_api = "assemblyai"
            result = parse_assemblyai_format(data)
        # ... Add elif blocks for other APIs ...
        # elif "words" in data and any(word.get("type") == "spacing" for word in data.get("words", [])):
        #     detected_api = "elevenlabs"
        #     result = parse_elevenlabs_format(data) 
        # elif "model" in data and data.get("model", "").startswith("whisper-"):
        #     detected_api = "openai"
        #     result = parse_openai_format(data)
        # elif "text" in data and "words" in data:
        #     # This is a bit generic, assume Groq or handle later
        #     detected_api = "groq" 
        #     result = parse_groq_format(data)
        else:
            # Fallback if detection fails
            text = data.get("text", "")
            if text:
                logger.warning("Could not determine JSON format, generating words from text")
                words = generate_words_from_text(text)
                result = TranscriptionResult(
                    text=text,
                    words=words,
                    language=data.get("language", ""),
                    api_name=detected_api
                )
            else:
                logger.warning("Could not determine JSON format or extract useful content")
                words = data.get("words", []) # Pass through if exists
                result = TranscriptionResult(
                    text=data.get("text", ""),
                    words=words,
                    language=data.get("language", ""),
                    api_name=detected_api
                )

    if result is None or (hasattr(result, 'words') and not result.words):
         logger.warning(f"Parsing resulted in an empty word list for detected API: {detected_api}")

    return detected_api, result


def load_json_data(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Loads JSON data from a file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Loaded JSON data as a dictionary, or None if loading fails.
    """
    logger.info(f"Loading JSON data from file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        logger.error(f"JSON file not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON file {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {e}")
        return None 