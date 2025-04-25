"""
Output formatting functions for transcription results
"""
import os
import re
import json
import requests
import textwrap
from datetime import timedelta
from pathlib import Path
from typing import Union, List, Dict, Any, Optional
from loguru import logger


def format_time(seconds: float) -> str:
    """
    Format time in seconds to SRT timestamp format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string in HH:MM:SS,mmm format
    
    From: elevenlabs - Format seconds for SRT timestamp
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds)
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def format_timedelta(td: timedelta) -> str:
    """
    Convert timedelta to a formatted string for SRT.
    
    Args:
        td: Timedelta object
        
    Returns:
        Formatted string in HH:MM:SS,mmm format
    
    From: diarization - Convert timedelta to formatted string
    """
    total_seconds = td.total_seconds()
    return format_time(total_seconds)


def create_srt(words: List[Dict[str, Any]], output_file: Union[str, Path], 
               chars_per_line: int = 80, silentportions: int = 0) -> None:
    """
    Create SRT file from words data.
    
    Args:
        words: List of word dictionaries with timing info
        output_file: Path to output SRT file
        chars_per_line: Maximum characters per subtitle line
        silentportions: Minimum duration in ms to mark silent portions
    
    From: elevenlabs - Create SRT file from word data
    """
    logger.info(f"Creating SRT file: {output_file}")
        
    with open(output_file, 'w', encoding='utf-8') as f:
        counter = 1
        current_text = ""
        current_start = None
        current_end = None
        
        # Handle initial silence
        if words and 'type' in words[0] and words[0]['type'] == 'word' and words[0]['start'] > 0 and silentportions > 0:
            f.write(f"{counter}\n")
            f.write(f"00:00:00,000 --> {format_time(words[0]['start'])}\n")
            f.write("(...)\n\n")
            counter += 1
        
        for word in words:
            # Skip empty or None entries
            if not word:
                continue
                
            # Check if the word dictionary has a 'type' key
            word_type = word.get('type')
            
            if word_type == 'spacing' and silentportions > 0:
                duration_ms = (word['end'] - word['start']) * 1000
                if duration_ms >= silentportions:
                    # Write current segment if exists
                    if current_text:
                        f.write(f"{counter}\n")
                        f.write(f"{format_time(current_start)} --> {format_time(current_end)}\n")
                        f.write(f"{current_text.strip()}\n\n")
                        counter += 1
                        current_text = ""
                    
                    # Write silent portion
                    f.write(f"{counter}\n")
                    f.write(f"{format_time(word['start'])} --> {format_time(word['end'])}\n")
                    f.write("(...)\n\n")
                    counter += 1
                    current_start = None
                    current_end = None
                    continue
            
            elif word_type == 'word':
                if current_start is None:
                    current_start = word['start']
                current_end = word['end']
                current_text += word['text'] + " "
                
                if len(current_text.strip()) >= chars_per_line:
                    f.write(f"{counter}\n")
                    f.write(f"{format_time(current_start)} --> {format_time(current_end)}\n")
                    f.write(f"{current_text.strip()}\n\n")
                    counter += 1
                    current_text = ""
                    current_start = None
                    current_end = None
            elif word_type == 'audio_event':
                # Write current segment if exists
                if current_text:
                    f.write(f"{counter}\n")
                    f.write(f"{format_time(current_start)} --> {format_time(current_end)}\n")
                    f.write(f"{current_text.strip()}\n\n")
                    counter += 1
                    current_text = ""
                    current_start = None
                    current_end = None
                
                # Write audio event as its own subtitle
                f.write(f"{counter}\n")
                f.write(f"{format_time(word['start'])} --> {format_time(word['end'])}\n")
                f.write(f"({word['text']})\n\n")
                counter += 1
            elif word_type is None:
                # Handle words without type key (assume they are regular words)
                if current_start is None:
                    current_start = word['start']
                current_end = word['end']
                current_text += word.get('text', '') + " "
                
                if len(current_text.strip()) >= chars_per_line:
                    f.write(f"{counter}\n")
                    f.write(f"{format_time(current_start)} --> {format_time(current_end)}\n")
                    f.write(f"{current_text.strip()}\n\n")
                    counter += 1
                    current_text = ""
                    current_start = None
                    current_end = None
        
        if current_text:
            f.write(f"{counter}\n")
            f.write(f"{format_time(current_start)} --> {format_time(current_end)}\n")
            f.write(f"{current_text.strip()}\n\n")
    logger.info("SRT file created successfully")


def process_davinci_block(file_obj, counter: int, block_words: List[Dict[str, Any]],
                          start_time: float, end_time: float) -> None:
    """
    Process a block of words for Davinci Resolve SRT format.
    Helper function for create_davinci_srt.
    
    Args:
        file_obj: File object to write to
        counter: Current subtitle counter
        block_words: List of words for this block
        start_time: Start time for this block
        end_time: End time for this block
    """
    if not block_words:
        return
    
    # Check if we need to split the block
    MAX_DAVINCI_WORDS = 500
    if len(block_words) > MAX_DAVINCI_WORDS:
        # Find sentence boundaries to split at
        sentences = []
        current_sentence = []
        sentence_end_markers = ['.', '!', '?', ':', ';']
        
        for word in block_words:
            current_sentence.append(word)
            word_text = word.get('text', '')
            if word_text and word_text[-1] in sentence_end_markers:
                sentences.append(current_sentence)
                current_sentence = []
        
        # Add any remaining words as a sentence
        if current_sentence:
            sentences.append(current_sentence)
        
        # Write each sentence as a separate block
        current_start = start_time
        current_counter = counter
        
        for i, sentence in enumerate(sentences):
            if not sentence:
                continue
                
            # Filter out audio events from sentence text
            regular_words = [w for w in sentence if w.get('type') != 'audio_event']
            
            if not regular_words:
                continue
                
            sentence_text = " ".join([w['text'] for w in regular_words])
            sentence_end = regular_words[-1]['end']
            
            # Check for audio events within this sentence timeframe
            audio_events = [w for w in sentence if w.get('type') == 'audio_event']
            if audio_events:
                # Write just the text first
                file_obj.write(f"{current_counter}\n")
                file_obj.write(f"{format_time(current_start)} --> {format_time(sentence_end)}\n")
                file_obj.write(f"{sentence_text}\n\n")
                current_counter += 1
                
                # Then write each audio event separately
                for event in audio_events:
                    file_obj.write(f"{current_counter}\n")
                    file_obj.write(f"{format_time(event['start'])} --> {format_time(event['end'])}\n")
                    file_obj.write(f"({event['text']})\n\n")
                    current_counter += 1
            else:
                # No audio events, write text normally
                file_obj.write(f"{current_counter}\n")
                file_obj.write(f"{format_time(current_start)} --> {format_time(sentence_end)}\n")
                file_obj.write(f"{sentence_text}\n\n")
            
            current_start = sentence_end
    else:
        # Write the entire block
        # Filter out audio events from block text
        regular_words = [w for w in block_words if w.get('type') != 'audio_event']
        audio_events = [w for w in block_words if w.get('type') == 'audio_event']
        
        if regular_words:
            block_text = " ".join([w['text'] for w in regular_words])
            file_obj.write(f"{counter}\n")
            file_obj.write(f"{format_time(start_time)} --> {format_time(end_time)}\n")
            file_obj.write(f"{block_text}\n\n")
            counter_offset = 1
            
            # Write audio events separately if they exist
            for event in audio_events:
                file_obj.write(f"{counter + counter_offset}\n")
                file_obj.write(f"{format_time(event['start'])} --> {format_time(event['end'])}\n")
                file_obj.write(f"({event['text']})\n\n")
                counter_offset += 1
        elif audio_events:
            # Only audio events, no regular words
            for i, event in enumerate(audio_events):
                file_obj.write(f"{counter + i}\n")
                file_obj.write(f"{format_time(event['start'])} --> {format_time(event['end'])}\n")
                file_obj.write(f"({event['text']})\n\n")


def create_davinci_srt(words: List[Dict[str, Any]], output_file: Union[str, Path], 
                      silentportions: int = 0, padding: int = 30) -> None:
    """
    Create SRT file optimized for Davinci Resolve Studio.
    
    Args:
        words: List of word dictionaries with timing info
        output_file: Path to output SRT file
        silentportions: Minimum duration in ms to mark silent portions
        padding: Padding in ms to add to word end times
        
    From: elevenlabs - Create SRT optimized for DaVinci
    """
    # Import locally to avoid circular imports
    from .text_processing import process_filler_words, merge_consecutive_pauses
    
    logger.info(f"Creating Davinci Resolve optimized SRT file: {output_file}")
    
    # Default pause detection is 200ms if not specified
    pause_detection = silentportions if silentportions > 0 else 200
    
    # Pre-process words to identify filler words
    words = process_filler_words(words, pause_detection)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        counter = 1
        block_words = []
        block_start = None
        block_end = None
        
        # Handle initial silence
        if words and words[0].get('type') == 'word' and words[0]['start'] > 0:
            f.write(f"{counter}\n")
            f.write(f"00:00:00,000 --> {format_time(words[0]['start'])}\n")
            f.write("(...)\n\n")
            counter += 1
        
        i = 0
        MAX_DAVINCI_WORDS = 500  # Maximum words per subtitle block for davinci mode
        
        while i < len(words):
            word = words[i]
            
            # Skip empty or None entries
            if not word:
                i += 1
                continue
                
            # Check if the word dictionary has a 'type' key
            word_type = word.get('type')
            
            # If we find a spacing that exceeds our pause threshold
            if word_type == 'spacing':
                duration_ms = (word['end'] - word['start']) * 1000
                if duration_ms >= pause_detection:
                    # Process the accumulated block of words
                    if block_words:
                        process_davinci_block(f, counter, block_words, block_start, block_end)
                        counter += 1
                        block_words = []
                        block_start = None
                        block_end = None
                    
                    # Write silent portion
                    f.write(f"{counter}\n")
                    f.write(f"{format_time(word['start'])} --> {format_time(word['end'])}\n")
                    f.write("(...)\n\n")
                    counter += 1
                else:
                    # Add padding to the previous word if it exists
                    if block_words:
                        block_words[-1]['end'] += padding / 1000.0
                        block_end = block_words[-1]['end']
            
            # Add word to current block
            elif word_type == 'word':
                if block_start is None:
                    block_start = word['start']
                block_end = word['end']
                block_words.append(word)
                
                # If we've reached the maximum words limit, process the block
                if len(block_words) >= MAX_DAVINCI_WORDS:
                    process_davinci_block(f, counter, block_words, block_start, block_end)
                    counter += 1
                    block_words = []
                    block_start = None
                    block_end = None
            
            # Handle audio events
            elif word_type == 'audio_event':
                # Process the accumulated block of words
                if block_words:
                    process_davinci_block(f, counter, block_words, block_start, block_end)
                    counter += 1
                    block_words = []
                    block_start = None
                    block_end = None
                
                # Write audio event as its own subtitle
                f.write(f"{counter}\n")
                f.write(f"{format_time(word['start'])} --> {format_time(word['end'])}\n")
                f.write(f"({word['text']})\n\n")
                counter += 1
            
            # Handle words without type key (assume they are regular words)
            elif word_type is None:
                if block_start is None:
                    block_start = word['start']
                block_end = word['end']
                block_words.append(word)
                
                # If we've reached the maximum words limit, process the block
                if len(block_words) >= MAX_DAVINCI_WORDS:
                    process_davinci_block(f, counter, block_words, block_start, block_end)
                    counter += 1
                    block_words = []
                    block_start = None
                    block_end = None
            
            i += 1
        
        # Process any remaining words
        if block_words:
            process_davinci_block(f, counter, block_words, block_start, block_end)
    
    # Post-process the SRT file to merge consecutive pause entries
    merge_consecutive_pauses(output_file)
    
    logger.info("Davinci Resolve SRT file created successfully")


def create_text_file(words: List[Dict[str, Any]], output_file: Union[str, Path]) -> None:
    """
    Create plain text file from words data.
    
    Args:
        words: List of word dictionaries with timing info
        output_file: Path to output text file
        
    From: elevenlabs - Create plain text transcript
    """
    logger.info(f"Creating text file: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        current_speaker = None
        for word in words:
            # Skip empty or None entries
            if not word:
                continue
                
            # Check if the word dictionary has a 'type' key
            word_type = word.get('type')
            
            if word_type is None:
                # Handle words without type key (assume they are regular words)
                speaker = word.get('speaker_id', 'Unknown')
                if speaker != current_speaker:
                    if current_speaker is not None:
                        f.write("\n")
                    f.write(f"Speaker {speaker}: ")
                    current_speaker = speaker
                f.write(word.get('text', '') + " ")
            elif word_type == 'word':
                speaker = word.get('speaker_id', 'Unknown')
                if speaker != current_speaker:
                    if current_speaker is not None:
                        f.write("\n")
                    f.write(f"Speaker {speaker}: ")
                    current_speaker = speaker
                f.write(word['text'] + " ")
            elif word_type == 'audio_event':
                f.write(f"({word['text']}) ")
    logger.info("Text file created successfully")


def convert_to_srt(result: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """
    Convert Groq's verbose JSON output to SRT format with metadata-based filtering.
    
    Args:
        result: Transcription result dictionary from Groq API
        output_path: Path to save the SRT file
        
    From: groq - Convert JSON transcript to SRT
    """
    def split_text_into_chunks(text: str, max_chars: int = 80) -> List[str]:
        """Split text into chunks of maximum length while respecting word boundaries"""
        return textwrap.wrap(text, width=max_chars, break_long_words=False)

    # Filter segments based on metadata quality indicators
    filtered_segments = []
    for segment in result['segments']:
        # Skip segments with quality issues
        no_speech_prob = segment.get('no_speech_prob', 0)
        avg_logprob = segment.get('avg_logprob', -0.5)
        compression_ratio = segment.get('compression_ratio', 1.0)
        start_time = segment.get('start', 0)
        
        # Only add segments meeting quality criteria
        if (no_speech_prob < 0.5 and       # Less than 50% chance of being non-speech
            avg_logprob > -0.5 and         # Better than -0.5 log probability
            0.8 < compression_ratio < 2.0 and  # Normal speech patterns
            (start_time != 0 or all(s.get('start', 0) == 0 for s in result['segments']))):
            filtered_segments.append(segment)

    # Sort segments by start time to ensure proper ordering
    filtered_segments.sort(key=lambda x: x.get('start', 0))

    # Merge overlapping segments
    merged_segments = []
    if filtered_segments:
        current_segment = filtered_segments[0].copy()
        
        for next_segment in filtered_segments[1:]:
            # If segments overlap or are very close (within 0.1s), merge them
            if next_segment['start'] <= current_segment['end'] + 0.1:
                current_segment['end'] = max(current_segment['end'], next_segment['end'])
                current_segment['text'] = current_segment['text'] + ' ' + next_segment['text']
            else:
                merged_segments.append(current_segment)
                current_segment = next_segment.copy()
        
        merged_segments.append(current_segment)

    srt_lines = []
    subtitle_index = 1

    for segment in merged_segments:
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text'].strip()

        if not text:  # Skip empty segments
            continue

        chunks = split_text_into_chunks(text)
        
        if len(chunks) == 1:
            srt_lines.append(f"{subtitle_index}")
            srt_lines.append(f"{format_time(start_time)} --> {format_time(end_time)}")
            srt_lines.append(chunks[0])
            srt_lines.append("")  # Empty line
            subtitle_index += 1
        else:
            # Distribute chunks evenly across segment duration
            chunk_duration = (end_time - start_time) / len(chunks)
            for i, chunk in enumerate(chunks):
                chunk_start = start_time + i * chunk_duration
                chunk_end = chunk_start + chunk_duration
                srt_lines.append(f"{subtitle_index}")
                srt_lines.append(f"{format_time(chunk_start)} --> {format_time(chunk_end)}")
                srt_lines.append(chunk)
                srt_lines.append("")  # Empty line
                subtitle_index += 1

    # Write SRT file
    output_path = Path(output_path)
    srt_path = output_path.with_suffix('.srt')
    with open(srt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(srt_lines))
    
    logger.info(f"SRT file saved to: {srt_path}")


def format_transcript_with_speakers(segments: List[Dict[str, Any]]) -> str:
    """
    Format transcript with speaker labels.
    
    Args:
        segments: List of transcript segments with speaker info
        
    Returns:
        Formatted transcript text with speaker labels
        
    From: assemblyai/diarization - Format transcript with speaker labels
    """
    transcript = ""
    current_speaker = None
    
    for segment in segments:
        speaker = segment.get('speaker', segment.get('speaker_id', 'Unknown'))
        text = segment.get('text', '')
        
        if not text.strip():
            continue
            
        if speaker != current_speaker:
            if current_speaker is not None:
                transcript += "\n\n"
            transcript += f"Speaker {speaker}: {text}"
            current_speaker = speaker
        else:
            transcript += f" {text}"
    
    return transcript


def export_subtitles(transcript_id: str, headers: Dict[str, str], 
                     subtitle_format: str, file_name: str) -> str:
    """
    Export subtitles using AssemblyAI API.
    
    Args:
        transcript_id: ID of the transcript
        headers: Headers for API request, including Auth
        subtitle_format: Either 'srt' or 'vtt'
        file_name: Base name for output file (without extension)
        
    Returns:
        Filename of saved subtitle file
        
    From: assemblyai - Export subtitles in specified format
    """
    # Set the API endpoint for exporting subtitles
    url = f"https://api.assemblyai.com/v2/transcript/{transcript_id}/{subtitle_format}"

    # Send a GET request to the API to get the subtitle data
    response = requests.get(url, headers=headers)

    # Check if the response status code is successful (200)
    if response.status_code == 200:
        # Save the subtitle data to a local file
        filename = f"{file_name}.{subtitle_format}"
        with open(filename, "wb") as subtitle_file:
            subtitle_file.write(response.content)
        return filename
    else:
        raise RuntimeError(f"Subtitle export failed: {response.text}")
