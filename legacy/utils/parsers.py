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
                segments: List[Dict[str, Any]] = None, api_name: str = "unknown"):
        """
        Initialize a TranscriptionResult.
        
        Args:
            text: Full transcript text
            confidence: Overall confidence score (0.0-1.0)
            language: Language code (ISO-639-1 or ISO-639-3)
            words: List of word dictionaries with timing info
            speakers: List of speaker dictionaries with identification info
            segments: List of segment dictionaries with timing info
            api_name: Name of the API used for transcription
        """
        self.text = text
        self.confidence = confidence
        self.language = language
        self.words = words or []
        self.speakers = speakers or []
        self.segments = segments or []
        self.api_name = api_name
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the transcription result to a dictionary."""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "language": self.language,
            "api_name": self.api_name,
            "words": self.words,
            "speakers": self.speakers,
            "segments": self.segments
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
            "segments": self.segments,
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
                                    # Format floating point numbers with fixed precision (3 decimal places)
                                    # This preserves millisecond precision in timestamps
                                    yield f'      "{k}": {format(v, ".3f")}'
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
                logger.debug(f"Saving transcription result with custom JSON encoder to preserve decimal precision")
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
                        yield "{\n  \"value\": [\n"
                        
                        for i, word in enumerate(obj):
                            word_copy = word.copy()
                            
                            # Ensure speaker_id is a simple string
                            if "speaker_id" in word_copy and (word_copy["speaker_id"] is None or word_copy["speaker_id"] == "Unknown"):
                                word_copy["speaker_id"] = ""
                            
                            yield "    {\n"
                            
                            word_items = list(word_copy.items())
                            for j, (k, v) in enumerate(word_items):
                                if k in ["start", "end"] and isinstance(v, (int, float)):
                                    # Format floating point numbers with fixed precision to avoid scientific notation
                                    # Keep decimal places for all timestamps to preserve precision
                                    yield f'      "{k}": {format(v, ".3f")}'
                                else:
                                    # Use standard JSON encoding for other fields
                                    yield f'      "{k}": '
                                    yield from super().iterencode(v)
                                
                                if j < len(word_items) - 1:
                                    yield ",\n"
                                else:
                                    yield "\n"
                            
                            if i < len(obj) - 1:
                                yield "    },\n"
                            else:
                                yield "    }\n"
                        
                        # Close the array
                        yield "  ],\n  \"Count\": " + str(len(obj)) + "\n}"
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
            segments=data.get("segments", []),
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
    
    # Check if timestamps are in milliseconds format by examining the first few words
    # If most values are large (>100), they're likely milliseconds
    ms_format_detected = False
    ms_count = 0
    s_count = 0
    
    # Sample the first 10 words (or all if fewer) to check timestamp format
    for word in raw_words[:min(10, len(raw_words))]:
        start = word.get('start', 0)
        end = word.get('end', 0)
        
        # Skip zero values
        if start == 0 and end == 0:
            continue
        
        # If value is >100, likely milliseconds
        if start > 100 or end > 100:
            ms_count += 1
        else:
            s_count += 1
    
    # If most timestamps appear to be in milliseconds
    if ms_count > s_count:
        ms_format_detected = True
        logger.info(f"Detected millisecond timestamps in AssemblyAI data. Converting to seconds format.")
    
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
        
        # Convert milliseconds to seconds if we detected ms format
        if ms_format_detected:
            start = start / 1000.0
            end = end / 1000.0
        
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
        segments=[],
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
    language = data.get("language", data.get("language_code", ""))
    
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
            if not isinstance(word_data, dict):
                continue
            text_val = word_data.get("text", "")
            start_val = word_data.get("start", 0)
            end_val = word_data.get("end", 0)
            raw_type = word_data.get("type")

            # Normalize type: word | spacing | audio_event
            normalized_type = "word"
            txt_stripped = (text_val or "").strip()
            if raw_type == "spacing" or txt_stripped == "":
                normalized_type = "spacing"
            elif raw_type in ("audio_event", "event"):
                normalized_type = "audio_event"
            elif txt_stripped.startswith("(") and txt_stripped.endswith(")"):
                # Heuristic: bracketed token is likely an audio event
                normalized_type = "audio_event"

            word_entry = {
                "text": text_val,
                "start": float(start_val) if isinstance(start_val, (int, float)) else 0.0,
                "end": float(end_val) if isinstance(end_val, (int, float)) else 0.0,
                "type": normalized_type,
            }
            # Preserve speaker identifiers if present
            speaker_id = word_data.get("speaker_id") or word_data.get("speaker")
            if speaker_id:
                word_entry["speaker_id"] = speaker_id
                word_entry["speaker"] = speaker_id

            words.append(word_entry)
    else:
        # If no words data but text is available, generate simple word objects
        if text:
            logger.info("Generating words from text content since words array is empty")
            words = generate_words_from_text(text)
        else:
            logger.warning("No words data found in ElevenLabs format and no text available")
    
    # Sort words chronologically to ensure proper order of words and audio events
    words = sorted(words, key=lambda w: (w.get("start", 0), 0 if w.get("type") != "spacing" else 1))

    # Aggregate speakers if present
    speakers = []
    unique_speakers = []
    seen = set()
    for w in words:
        spk = w.get("speaker") or w.get("speaker_id")
        if spk and spk not in seen:
            seen.add(spk)
            unique_speakers.append({"id": spk})
    speakers = unique_speakers
    
    return TranscriptionResult(
        text=text,
        words=words,
        language=language,
        api_name="elevenlabs",
        speakers=speakers
    )


def parse_groq_format(data: Dict[str, Any]) -> TranscriptionResult:
    """
    Parse Groq transcription data into standardized format.
    If 'words' is empty but 'text' is present, generate word-level timings.
    
    Handles Groq's specific format with S.ms timestamps (e.g., 0.0, 0.5 seconds).
    """
    logger.debug("Parsing Groq format")
    text = data.get("text", "")
    language = data.get("language", "")
    
    # Log structure information for debugging
    logger.debug(f"Groq response keys: {list(data.keys())}")
    
    # Check for word-level timestamps in the response
    has_word_timestamps = False
    if "words" in data and isinstance(data["words"], list) and len(data["words"]) > 0:
        logger.debug(f"Found words array with {len(data['words'])} items")
        if len(data["words"]) > 0:
            logger.debug(f"First word item: {data['words'][0]}")
        has_word_timestamps = True
        raw_words = data["words"]
    else:
        logger.debug("No valid words array found in Groq response")
        # Try to extract words from segments if available
        if "segments" in data and isinstance(data["segments"], list) and len(data["segments"]) > 0:
            logger.debug(f"Found segments array with {len(data['segments'])} items")
            # If we have segments with words, extract them
            all_words = []
            for segment in data["segments"]:
                if isinstance(segment, dict) and "words" in segment and isinstance(segment["words"], list):
                    all_words.extend(segment["words"])
            
            if len(all_words) > 0:
                logger.debug(f"Extracted {len(all_words)} words from segments")
                has_word_timestamps = True
                raw_words = all_words
            else:
                logger.debug("No words found in segments")
                raw_words = []
        else:
            logger.debug("No segments found with word-level timestamps")
            raw_words = []

    # Fallback: generate word timings if missing
    if not has_word_timestamps and text:
        logger.info("Generating words from text content since words array is empty")
        tokens = text.split()
        fake_duration = 0.5  # seconds per word
        raw_words = []
        for i, token in enumerate(tokens):
            start_time = i * fake_duration
            end_time = start_time + fake_duration
            raw_words.append({
                "text": token,
                "start": start_time,
                "end": end_time,
                "confidence": 0.9  # Fake confidence score
            })

    # Process word list to handle Groq's unique S.ms format
    words_with_spacing = []
    
    # Sort words by start time to ensure correct order
    raw_words = sorted(raw_words, key=lambda w: float(w.get('start', 0)))
    
    # Debugging info
    logger.debug(f"Processing {len(raw_words)} words from Groq format")
    if raw_words and len(raw_words) > 0:
        logger.debug(f"First word: {raw_words[0]}")
    
    # Process words to add spacing elements and preserve decimal precision
    prev_end = None
    
    for word_data in raw_words:
        # Handle Groq's different word format (using 'word' instead of 'text')
        word_text = word_data.get('text', word_data.get('word', ''))
        
        # Ensure we have start and end
        if 'start' not in word_data or 'end' not in word_data:
            logger.warning(f"Skipping word data missing required keys: {word_data}")
            continue
            
        # Handle different time formats
        start = float(word_data.get('start', 0))
        end = float(word_data.get('end', 0))
        
        # Create word object
        word_obj = {
            "text": word_text,
            "start": start,
            "end": end,
            "type": "word"
        }
        
        # Only add confidence if present
        if 'confidence' in word_data:
            word_obj['confidence'] = word_data.get('confidence')
            
        # Add spacing element if this isn't the first word
        if prev_end is not None:
            # Calculate gap between words
            gap = start - prev_end
            
            # Only add spacing if there's a meaningful gap
            if gap > 0.001:  # Small threshold to avoid rounding issues
                spacing_obj = {
                    "text": "",
                    "start": prev_end,
                    "end": start,
                    "type": "spacing"
                }
                words_with_spacing.append(spacing_obj)
        
        words_with_spacing.append(word_obj)
        prev_end = end
    
    # Check if we added spacing elements
    word_count = sum(1 for w in words_with_spacing if w.get('type') == 'word')
    spacing_count = sum(1 for w in words_with_spacing if w.get('type') == 'spacing')
    logger.debug(f"Final result: {word_count} words and {spacing_count} spacing elements")
    
    # Create segments from words if needed
    segments = data.get("segments", [])
    if not segments and words_with_spacing:
        # Only use word elements (not spacing) for segment creation
        word_items = [w for w in words_with_spacing if w.get('type') == 'word']
        word_groups = [word_items[i:i + 10] for i in range(0, len(word_items), 10)]
        segments = []
        
        for i, group in enumerate(word_groups):
            if not group:
                continue
                
            segment_text = " ".join(word.get("text", "") for word in group)
            segment = {
                "id": i,
                "start": group[0].get("start", 0),
                "end": group[-1].get("end", 0),
                "text": segment_text
            }
            segments.append(segment)
    
    # Create result
    result = TranscriptionResult(
        text=text,
        language=language,
        words=words_with_spacing,
        segments=segments,
        api_name=data.get("api_name", "groq")
    )
    
    logger.debug(f"Created TranscriptionResult with {len(result.words)} words")
    return result


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
    
    # Check for words in verbose_json format
    if data.get("words") and isinstance(data["words"], list):
        logger.debug(f"Processing {len(data['words'])} words from OpenAI Whisper verbose JSON")
        
        for word_data in data["words"]:
            if isinstance(word_data, dict):
                # Handle OpenAI's word format which uses "word" instead of "text"
                word_text = word_data.get("word", word_data.get("text", ""))
                start = word_data.get("start", 0)  # OpenAI uses seconds
                end = word_data.get("end", 0)      # OpenAI uses seconds
                
                # If no end time is provided, estimate based on word length
                if end == 0 and start != 0:
                    # Rough estimate: 0.15s per character
                    word_length = len(word_text)
                    end = start + max(0.3, word_length * 0.15)
                
                word = {
                    "text": word_text,
                    "start": start,
                    "end": end,
                    "type": "word"
                }
                words.append(word)
    # If words is null in response but was requested, log warning
    elif data.get("words") is None and data.get("timestamp_granularities") == ["word"]:
        logger.warning("OpenAI API returned words: null despite requesting word timestamps. Check API version support.")
    elif "segments" in data and data["segments"]:
        # Extract words from segments if available or split segment text into words
        logger.debug("Words are null, extracting from segments or generating from text")
        
        for segment in data["segments"]:
            if isinstance(segment, dict) and "text" in segment:
                segment_text = segment.get("text", "").strip()
                segment_start = segment.get("start", 0.0)
                segment_end = segment.get("end", 0.0)
                
                if not segment_text:
                    continue
                
                # Split segment text into words and assign estimated timestamps
                segment_words = segment_text.split()
                if not segment_words:
                    continue
                
                # Calculate word durations proportionally within segment
                segment_duration = segment_end - segment_start
                total_chars = sum(len(word) for word in segment_words)
                
                current_time = segment_start
                for word_text in segment_words:
                    if not word_text:
                        continue
                    
                    # Calculate word duration proportionally by character count
                    word_duration = (len(word_text) / total_chars) * segment_duration if total_chars > 0 else 0.3
                    word_duration = max(0.1, word_duration)  # Minimum word duration
                    
                    word = {
                        "text": word_text,
                        "start": current_time,
                        "end": current_time + word_duration,
                        "type": "word"
                    }
                    words.append(word)
                    
                    # Add spacing after word
                    spacing_duration = 0.05  # 50ms spacing between words
                    space = {
                        "text": " ",
                        "start": current_time + word_duration,
                        "end": current_time + word_duration + spacing_duration,
                        "type": "spacing"
                    }
                    words.append(space)
                    
                    current_time += word_duration + spacing_duration
    else:
        # If no words data but text is available, generate simple word objects
        if text:
            logger.info("Generating words from text content since words array is empty")
            words = generate_words_from_text(text)
        else:
            logger.warning("No words data found in OpenAI format and no text available")
    
    # Add spacing elements between words if they don't exist and weren't already added above
    words_with_spacing = []
    has_spacing = any(word.get("type") == "spacing" for word in words)
    
    if has_spacing:
        # Already has spacing elements, just use the words list
        words_with_spacing = words
    else:
        # Need to add spacing between words
        for i, word in enumerate(words):
            if word.get("type") == "spacing":
                # Skip if it's already a spacing element
                continue
                
            words_with_spacing.append(word)
            
            # Add spacing after word (except for the last word)
            if i < len(words) - 1:
                next_word = words[i + 1]
                if next_word.get("type") != "spacing":
                    space = {
                        "text": " ",
                        "start": word["end"],
                        "end": next_word["start"] if next_word["start"] > word["end"] else word["end"] + 0.1,
                        "type": "spacing"
                    }
                    words_with_spacing.append(space)
    
    # Create the standardized result
    result = TranscriptionResult(
        text=text,
        words=words_with_spacing,
        language=language,
        api_name="openai",
        speakers=[]
    )
    
    return result


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


def detect_and_parse_json(data: Dict[str, Any]) -> Tuple[str, TranscriptionResult]:
    """
    Auto-detect the JSON format and parse it into a TranscriptionResult object.
    
    Args:
        data: JSON data from any supported API
        
    Returns:
        Tuple containing detected api_name (str) and a TranscriptionResult object
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
        elif api_name == "elevenlabs":
            result = parse_elevenlabs_format(data) 
        elif api_name == "groq":
            result = parse_groq_format(data)
        elif api_name == "openai":
            result = parse_openai_format(data)
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
        elif "words" in data and any(word.get("type") == "spacing" for word in data.get("words", [])):
            detected_api = "elevenlabs"
            result = parse_elevenlabs_format(data) 
        elif "model" in data and data.get("model", "").startswith("whisper-"):
            detected_api = "openai"
            result = parse_openai_format(data)
        elif "segments" in data and any("no_speech_prob" in segment for segment in data.get("segments", [])):
            # Groq uses segments with no_speech_prob from Whisper
            detected_api = "groq" 
            result = parse_groq_format(data)
        elif "text" in data and "words" in data and all(isinstance(w.get("confidence", 0), (int, float)) for w in data.get("words", [])[:5]):
            # This matches Groq's simple word format with 0.5s intervals
            detected_api = "groq" 
            result = parse_groq_format(data)
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

    if result is None:
        # Create an empty result if all else fails
        logger.warning(f"Parsing failed to produce a valid result for detected API: {detected_api}")
        result = TranscriptionResult(
            text=data.get("text", ""),
            words=[],
            language=data.get("language", ""),
            api_name=detected_api
        )
    elif not result.words:
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


def parse_json_by_api(data: Dict[str, Any], api_name: str) -> Optional[TranscriptionResult]:
    """
    Parse JSON data according to the specified API format.
    
    Args:
        data: Raw JSON data from the API
        api_name: Name of the API ('assemblyai', 'elevenlabs', 'groq', 'openai')
        
    Returns:
        Standardized TranscriptionResult object or None if parsing fails
    """
    if not data:
        logger.error("No data provided for parsing")
        return None
        
    api_name = api_name.lower()
    
    try:
        if api_name == "assemblyai":
            return parse_assemblyai_format(data)
        elif api_name == "elevenlabs":
            return parse_elevenlabs_format(data)
        elif api_name == "groq":
            return parse_groq_format(data)
        elif api_name == "openai":
            return parse_openai_format(data)
        else:
            # Try to detect the API type from the data structure
            detected_api, result = detect_and_parse_json(data)
            logger.info(f"Auto-detected API format: {detected_api}")
            return result
    except Exception as e:
        logger.error(f"Error parsing {api_name} format: {str(e)}")
        return None 