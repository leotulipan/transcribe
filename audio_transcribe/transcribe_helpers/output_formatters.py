"""
Output formatting functions for transcription results
"""
import os
import re
import io
import math
import json
import requests
import textwrap
from datetime import timedelta, datetime
from pathlib import Path
from typing import Union, List, Dict, Any, Optional
import copy

# Try to import loguru, fallback to our mock implementation
try:
    from loguru import logger
except ImportError:
    import sys
    import os
    
    # Add the parent directory to sys.path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Import our mock logger
    from loguru_patch import logger

# Import process_filler_words from text_processing module
from audio_transcribe.transcribe_helpers.text_processing import process_filler_words


def join_text_with_proper_spacing(current_text: str, new_word: str) -> str:
    """
    Join text with proper spacing after periods, commas, etc.
    
    Args:
        current_text: The current text string
        new_word: The new word to add
        
    Returns:
        The joined text with proper spacing
    """
    if not current_text:
        return new_word
    
    # Check if the current text ends with punctuation that needs a space after it
    punctuation_needing_space = ".,:;!?"
    
    # Get the last character of current text
    last_char = current_text[-1]
    
    # If the last character is punctuation that needs a space, add one
    if last_char in punctuation_needing_space:
        return current_text + " " + new_word
    # If the last character is not punctuation, add a space before the new word
    elif last_char not in ".,;:!?-":
        return current_text + " " + new_word
    else:
        # Last character is punctuation that doesn't need a space (like hyphen)
        return current_text + new_word


def format_time(seconds: float, start_hour: int = 0) -> str:
    """Format time in seconds to HH:MM:SS,ms format."""
    # Convert seconds to milliseconds integer for calculation
    total_milliseconds = int(round(seconds * 1000))
    
    hours, total_milliseconds = divmod(total_milliseconds, 3600000)
    hours += start_hour
    minutes, total_milliseconds = divmod(total_milliseconds, 60000)
    seconds_part, milliseconds = divmod(total_milliseconds, 1000)
    
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d},{milliseconds:03d}"


def format_time_ms(milliseconds: int, start_hour: int = 0) -> str:
    """Format time in integer milliseconds to HH:MM:SS,ms format."""
    if not isinstance(milliseconds, int):
        logger.warning(f"format_time_ms received non-integer: {milliseconds}. Converting.")
        milliseconds = int(round(milliseconds))
    
    # Sanity check for extremely large millisecond values that might indicate a scaling issue
    if milliseconds > 100000000:  # 100,000 seconds = ~28 hours
        logger.warning(f"Very large timestamp detected: {milliseconds}ms ({milliseconds/1000/60/60:.2f} hours). This may indicate a scaling problem.")
    
    # Process as normal
    hours, milliseconds = divmod(milliseconds, 3600000)
    hours += start_hour
    minutes, milliseconds = divmod(milliseconds, 60000)
    seconds_part, milliseconds = divmod(milliseconds, 1000)
    
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d},{milliseconds:03d}"


def retime_subtitles_fps(words: List[Dict[str, Any]], fps: float, 
                         fps_offset_start: int = -1, fps_offset_end: int = 0) -> List[Dict[str, Any]]:
    """
    Retime subtitles based on frame numbers.
    Negative offset moves time earlier, positive moves later (for both start and end).
    
    Args:
        words: List of word dictionaries with timing info
        fps: Frames per second
        fps_offset_start: Frames to offset from start time (default -1)
        fps_offset_end: Frames to offset from end time (default 0)
        
    Returns:
        List of word dictionaries with retimed start and end values
    """
    logger.info(f"Retiming subtitles with FPS={fps}, offset_start={fps_offset_start}, offset_end={fps_offset_end}")
    
    retimed_words = []
    for word in words:
        if not word:
            retimed_words.append(word)
            continue
            
        # Only process words with start/end times
        if 'start' in word and 'end' in word:
            word_copy = word.copy()
            
            # Convert start time to frames, apply offset, convert back to seconds
            start_frames = int(word['start'] * fps)
            adjusted_start_frames = start_frames + fps_offset_start
            word_copy['start'] = adjusted_start_frames / fps
            
            # Convert end time to frames, apply offset, convert back to seconds
            end_frames = int(word['end'] * fps)
            adjusted_end_frames = end_frames + fps_offset_end
            word_copy['end'] = adjusted_end_frames / fps
            
            retimed_words.append(word_copy)
        else:
            retimed_words.append(word)
            
    return retimed_words


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
               chars_per_line: int = 80, silentportions: int = 0,
               fps: Optional[float] = None, fps_offset_start: int = -1, 
               fps_offset_end: int = 0, padding_start: int = 0, padding_end: int = 0,
               srt_mode: str = "standard", max_words_per_block: int = 0,
               remove_fillers: bool = False, filler_words: Optional[List[str]] = None,
               show_pauses: bool = False, start_hour: int = 0,
               words_per_subtitle: int = 0, filler_lines: bool = False) -> None:
    """
    Create SRT subtitle file from words.
    
    Args:
        words: List of word objects with timing information
        output_file: Path to output SRT file
        chars_per_line: Maximum characters per line
        silentportions: Milliseconds threshold for silence
        fps: Frames per second for frame-based timing
        fps_offset_start: Frame offset for start time
        fps_offset_end: Frame offset for end time
        padding_start: Milliseconds to adjust word start times
        padding_end: Milliseconds to adjust word end times
        srt_mode: "standard", "word", or "davinci"
        max_words_per_block: Max words per subtitle block (0=no limit)
        remove_fillers: Whether to remove filler words
        filler_words: List of filler words to remove
        show_pauses: Whether to show pause indicators (...)
        start_hour: Hour to offset timestamps
    """
    # Log all subtitle settings
    logger.debug("SRT Generation Settings:")
    logger.debug(f"- Output file: {output_file}")
    logger.debug(f"- SRT mode: {srt_mode}")
    logger.debug(f"- Characters per line: {chars_per_line}")
    if words_per_subtitle and words_per_subtitle > 0:
        logger.debug(f"- Words per subtitle: {words_per_subtitle}")
    logger.debug(f"- Silent threshold: {silentportions}ms")
    logger.debug(f"- Padding start: {padding_start}ms")
    logger.debug(f"- Padding end: {padding_end}ms")
    logger.debug(f"- Show pauses: {show_pauses}")
    logger.debug(f"- Speaker labels: {True}")
    if fps:
        logger.debug(f"- FPS: {fps}")
        logger.debug(f"- FPS offset start: {fps_offset_start} frames")
        logger.debug(f"- FPS offset end: {fps_offset_end} frames")
    logger.debug(f"- Remove fillers: {remove_fillers}")
    logger.debug(f"- Filler lines: {filler_lines}")
    
    # Count pause markers in input
    pause_marker_count = sum(1 for w in words if w.get('type') == 'spacing' and '(...)' in w.get('text', ''))
    logger.debug(f"Input words already contain {pause_marker_count} pause markers")
    
    # Log first 5 words for debugging
    if len(words) > 0:
        logger.debug(f"First 5 words: {words[:5]}")
    
    # Make a copy of the words list to avoid modifying the original
    words_copy = copy.deepcopy(words)
    
    # Process filler words if needed (disabled when filler_lines is True)
    if remove_fillers:
        if filler_words is None:
            filler_words = ["äh", "ähm", "ah", "ahm", "uh", "er", "hm", "hmm"]
            
        words_copy = process_filler_words(words_copy, silentportions, filler_words)
    
    # If filler_lines requested, transform filler words into standalone entries and uppercase them
    if filler_lines:
        if filler_words is None:
            filler_words = ["äh", "ähm", "ah", "ahm", "uh", "er", "hm", "hmm"]
        filler_set = {fw.lower() for fw in filler_words}
        transformed: List[Dict[str, Any]] = []
        for i, word in enumerate(words_copy):
            if word and word.get('type') == 'word':
                raw_text = word.get('text', '')
                normalized = re.sub(r"\W+", "", raw_text, flags=re.UNICODE).lower()
                if normalized in filler_set and raw_text.strip():
                    # Output filler as its own word-level subtitle later by marking as audio_event-like
                    # Convert to an audio_event to force separate subtitle lines downstream
                    filler_entry = word.copy()
                    filler_entry['text'] = raw_text.upper()
                    filler_entry['type'] = 'audio_event'
                    filler_entry['is_filler'] = True
                    transformed.append(filler_entry)
                    # Surrounding spacings remain as-is; we do not remove word from flow
                    continue
            transformed.append(word)
        words_copy = transformed
        
    # Make sure pauses are properly detected and marked (only if not already done)
    if show_pauses and silentportions > 0 and pause_marker_count == 0:
        from audio_transcribe.transcribe_helpers.text_processing import standardize_word_format
        logger.debug(f"Re-standardizing words after filler removal to detect pauses: show_pauses={show_pauses}, silence_threshold={silentportions}")
        words_copy = standardize_word_format(
            words_copy, 
            show_pauses=show_pauses, 
            silence_threshold=silentportions
        )
        # Count pause markers after standardization
        pause_marker_count = sum(1 for w in words_copy if w.get('type') == 'spacing' and '(...)' in w.get('text', ''))
        logger.debug(f"After re-standardization: {pause_marker_count} pause markers")
    
    # Apply padding to word timings if requested
    if padding_start != 0 or padding_end != 0:
        apply_intelligent_padding(words_copy, padding_start, padding_end)
    
    # Handle FPS-based timing if requested
    if fps is not None:
        words_copy = retime_subtitles_fps(words_copy, fps, fps_offset_start, fps_offset_end)
    
    # Choose SRT creation mode
    if srt_mode == "word":
        create_word_level_srt(words_copy, output_file, fps=fps, padding_start=0, padding_end=0, start_hour=start_hour)
    elif srt_mode == "davinci":
        create_davinci_srt(
            words_copy, output_file, silentportions=silentportions,
            fps=fps, padding_start=0, padding_end=0, remove_fillers=False, start_hour=start_hour
        )
    else:  # standard
        create_standard_srt(
            words_copy, output_file, chars_per_line=chars_per_line,
            silentportions=silentportions, fps=fps, padding_start=0, padding_end=0,
            words_per_subtitle=words_per_subtitle, start_hour=start_hour
        )
    
    logger.info(f"Created SRT file: {output_file}")


def apply_intelligent_padding(words: List[Dict[str, Any]], padding_start: int, padding_end: int) -> None:
    """
    Apply padding to word timestamps, but only into spacing gaps.
    - padding_start: move start time earlier (if negative) or later (if positive)
    - padding_end: move end time earlier (if negative) or later (if positive)
    
    Only applies padding if there's an adjacent spacing block and limits padding
    to the available space in that block. Also adjusts spacing block times to
    prevent overlaps.
    
    Args:
        words: List of word dictionaries with timing info
        padding_start: Milliseconds to offset start times
        padding_end: Milliseconds to offset end times
    """
    if not words or (padding_start == 0 and padding_end == 0):
        return
    
    logger.info(f"Applying intelligent padding: start={padding_start}ms, end={padding_end}ms")
    
    # Loop through words to apply padding
    for i, word in enumerate(words):
        if not word or word.get('type') == 'spacing':
            continue
        
        # Handle padding_start (looking at previous word)
        if padding_start != 0 and i > 0 and 'start' in word:
            prev_word = words[i-1]
            if prev_word and prev_word.get('type') == 'spacing':
                padding_seconds = padding_start / 1000.0
                original_start = word['start']
                
                # For negative padding (move start time earlier)
                if padding_start < 0:
                    # Calculate available space (limited by spacing start)
                    max_padding = word['start'] - prev_word['start']
                    # Apply limited padding
                    applied_padding = max(padding_seconds, -max_padding)
                    word['start'] += applied_padding
                    
                    # Prevent overlaps by adjusting previous spacing end time
                    if word['start'] < prev_word['end']:
                        prev_word['end'] = word['start']
                    
                    # logger.debug(f"Word {i}: Applied start padding {applied_padding*1000:.1f}ms (requested {padding_start}ms)")
                
                # For positive padding (move start time later)
                elif padding_start > 0:
                    # Limit by spacing duration
                    spacing_duration = prev_word['end'] - prev_word['start']
                    max_padding = min(spacing_duration, (word['start'] - prev_word['start']))
                    applied_padding = min(padding_seconds, max_padding)
                    word['start'] += applied_padding
                    logger.debug(f"Word {i}: Applied start padding {applied_padding*1000:.1f}ms (requested {padding_start}ms)")
        
        # Handle padding_end (looking at next word)
        if padding_end != 0 and i < len(words) - 1 and 'end' in word:
            next_word = words[i+1]
            if next_word and next_word.get('type') == 'spacing':
                padding_seconds = padding_end / 1000.0
                original_end = word['end']
                
                # For negative padding (move end time earlier)
                if padding_end < 0:
                    # Calculate available space (limited by current word duration)
                    word_duration = word['end'] - word['start']
                    # Ensure we don't make the word too short
                    min_duration = 0.05  # 50ms minimum
                    max_padding = word_duration - min_duration
                    # Apply limited padding
                    applied_padding = max(padding_seconds, -max_padding)
                    word['end'] += applied_padding
                    logger.debug(f"Word {i}: Applied end padding {applied_padding*1000:.1f}ms (requested {padding_end}ms)")
                
                # For positive padding (move end time later)
                elif padding_end > 0:
                    # Limit by next spacing end
                    max_padding = next_word['end'] - word['end']
                    applied_padding = min(padding_seconds, max_padding)
                    word['end'] += applied_padding
                    
                    # Prevent overlaps by adjusting next spacing start time
                    if word['end'] > next_word['start']:
                        next_word['start'] = word['end']
                    
                    logger.debug(f"Word {i}: Applied end padding {applied_padding*1000:.1f}ms (requested {padding_end}ms)")
    
    # Additional pass to ensure no overlaps between any segments
    for i in range(1, len(words)):
        if words[i-1] and words[i] and 'end' in words[i-1] and 'start' in words[i]:
            if words[i]['start'] < words[i-1]['end']:
                # Use a small buffer (1ms) to absolutely ensure no overlap
                middle_point = (words[i-1]['end'] + words[i]['start']) / 2
                words[i-1]['end'] = middle_point - 0.001
                words[i]['start'] = middle_point + 0.001
                logger.debug(f"Fixed overlap between words {i-1} and {i}")


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
    # Ensure timestamp formatting uses a defined start_hour (defaults to 0)
    start_hour = 0
    
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
                
            # Join words with proper spacing after punctuation
            sentence_text = ""
            for word in regular_words:
                sentence_text = join_text_with_proper_spacing(sentence_text, word['text'])
            sentence_end = regular_words[-1]['end']
            
            # Check for audio events within this sentence timeframe
            audio_events = [w for w in sentence if w.get('type') == 'audio_event']
            if audio_events:
                # Write just the text first
                start_ms = int(current_start * 1000)
                end_ms = int(sentence_end * 1000)
                
                file_obj.write(f"{current_counter}\n")
                file_obj.write(f"{format_time_ms(start_ms, start_hour)} --> {format_time_ms(end_ms, start_hour)}\n")
                file_obj.write(f"{sentence_text}\n\n")
                current_counter += 1
                
                # Then write each audio event separately
                for event in audio_events:
                    start_ms = int(event['start'] * 1000)
                    end_ms = int(event['end'] * 1000)
                    
                    file_obj.write(f"{current_counter}\n")
                    file_obj.write(f"{format_time_ms(start_ms, start_hour)} --> {format_time_ms(end_ms, start_hour)}\n")
                    if event.get('is_filler'):
                        file_obj.write(f"{event['text'].upper()}\n\n")
                    else:
                        file_obj.write(f"({event['text']})\n\n")
                    current_counter += 1
            else:
                # No audio events, write text normally
                start_ms = int(current_start * 1000)
                end_ms = int(sentence_end * 1000)
                
                file_obj.write(f"{current_counter}\n")
                file_obj.write(f"{format_time_ms(start_ms, start_hour)} --> {format_time_ms(end_ms, start_hour)}\n")
                file_obj.write(f"{sentence_text}\n\n")
            
            current_start = sentence_end
    else:
        # Write the entire block
        # Filter out audio events from block text
        regular_words = [w for w in block_words if w.get('type') != 'audio_event']
        audio_events = [w for w in block_words if w.get('type') == 'audio_event']
        
        if regular_words:
            # Join words with proper spacing after punctuation
            block_text = ""
            for word in regular_words:
                block_text = join_text_with_proper_spacing(block_text, word['text'])
            
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            
            file_obj.write(f"{counter}\n")
            file_obj.write(f"{format_time_ms(start_ms, start_hour)} --> {format_time_ms(end_ms, start_hour)}\n")
            file_obj.write(f"{block_text}\n\n")
            counter_offset = 1
            
            # Write audio events separately if they exist
            for event in audio_events:
                start_ms = int(event['start'] * 1000)
                end_ms = int(event['end'] * 1000)
                
                file_obj.write(f"{counter + counter_offset}\n")
                file_obj.write(f"{format_time_ms(start_ms, start_hour)} --> {format_time_ms(end_ms, start_hour)}\n")
                if event.get('is_filler'):
                    file_obj.write(f"{event['text'].upper()}\n\n")
                else:
                    file_obj.write(f"({event['text']})\n\n")
                counter_offset += 1
        elif audio_events:
            # Only audio events, no regular words
            for i, event in enumerate(audio_events):
                start_ms = int(event['start'] * 1000)
                end_ms = int(event['end'] * 1000)
                
                file_obj.write(f"{counter + i}\n")
                file_obj.write(f"{format_time_ms(start_ms, start_hour)} --> {format_time_ms(end_ms, start_hour)}\n")
                if event.get('is_filler'):
                    file_obj.write(f"{event['text'].upper()}\n\n")
                else:
                    file_obj.write(f"({event['text']})\n\n")


def create_standard_srt(words: List[Dict[str, Any]], output_file: Union[str, Path], 
                       chars_per_line: int = 80, silentportions: int = 0,
                       fps: Optional[float] = None, fps_offset_start: int = -1, 
                       fps_offset_end: int = 0, padding_start: int = 0, padding_end: int = 0,
                       words_per_subtitle: int = 0, start_hour: int = 0) -> None:
    """Standard SRT format with character limits per line"""
    # Instead of recursively calling create_srt, implement the standard SRT logic directly here
    output_file = Path(output_file)
    counter = 1
    
    with open(output_file, 'w', encoding='utf-8') as file_obj:
        current_subtitle = []
        current_text = ""
        current_start = None
        current_end = None
        
        for i, word in enumerate(words):
            # Handle audio events as their own subtitles
            if word.get('type') == 'audio_event':
                # If we were building a text block, flush it first
                if current_subtitle and current_start is not None and current_end is not None and current_text.strip():
                    start_ms = int(current_start * 1000)
                    end_ms = int(current_end * 1000)
                    text_lines = textwrap.wrap(current_text, width=chars_per_line, break_long_words=False)
                    if not text_lines:
                        text_lines = [current_text]
                    file_obj.write(f"{counter}\n")
                    file_obj.write(f"{format_time_ms(start_ms, start_hour)} --> {format_time_ms(end_ms, start_hour)}\n")
                    file_obj.write("\n".join(text_lines) + "\n\n")
                    counter += 1
                    current_subtitle = []
                    current_text = ""
                    current_start = None
                    current_end = None

                # Now output the audio event as its own subtitle
                ev_text = word.get('text', '').strip()
                if ev_text:
                    start_ms = int(word.get('start', 0) * 1000)
                    end_ms = int(word.get('end', word.get('start', 0) + 0.5) * 1000)
                    file_obj.write(f"{counter}\n")
                    file_obj.write(f"{format_time_ms(start_ms, start_hour)} --> {format_time_ms(end_ms, start_hour)}\n")
                    # Ensure parentheses around event text
                    if not (ev_text.startswith('(') and ev_text.endswith(')')):
                        ev_text = f"({ev_text})"
                    file_obj.write(ev_text + "\n\n")
                    counter += 1
                continue
            if word.get('type') == 'spacing' and silentportions > 0:
                # Check if this is a meaningful silent portion
                if len(current_subtitle) > 0 and word.get('text', '').strip() and word.get('end', 0) - word.get('start', 0) >= silentportions / 1000.0:
                    # We have a significant pause, output current subtitle
                    if current_subtitle and current_start is not None and current_end is not None:
                        # Write current subtitle
                        start_ms = int(current_start * 1000)
                        end_ms = int(current_end * 1000)
                        
                        # Split text into lines based on character limit
                        text_lines = textwrap.wrap(current_text, width=chars_per_line, break_long_words=False)
                        if not text_lines:
                            text_lines = [current_text]  # Fallback in case wrap returned empty list
                        
                        file_obj.write(f"{counter}\n")
                        file_obj.write(f"{format_time_ms(start_ms, start_hour)} --> {format_time_ms(end_ms, start_hour)}\n")
                        file_obj.write("\n".join(text_lines) + "\n\n")
                        counter += 1
                    
                    # Reset for next subtitle
                    current_subtitle = []
                    current_text = ""
                    current_start = None
                    current_end = None
                    
                    # Add pause marker as its own subtitle if needed
                    if word.get('text', '').strip():
                        pause_start = int(word.get('start', 0) * 1000)
                        pause_end = int(word.get('end', 0) * 1000)
                        
                        file_obj.write(f"{counter}\n")
                        file_obj.write(f"{format_time_ms(pause_start, start_hour)} --> {format_time_ms(pause_end, start_hour)}\n")
                        file_obj.write(word.get('text', '').strip() + "\n\n")
                        counter += 1
                
                continue
            
            # Skip non-word items that aren't significant pauses
            if word.get('type') == 'spacing' or not word.get('text', '').strip():
                continue
            
            # Add this word to the current subtitle
            if not current_subtitle:
                # First word of this subtitle
                current_start = word.get('start', 0)
            
            current_subtitle.append(word)
            current_end = word.get('end', 0)
            
            # Join text with proper spacing after punctuation
            current_text = join_text_with_proper_spacing(current_text, word.get('text', '').strip())
            
            # Check if we should break subtitle here based on words_per_subtitle or character length
            should_break = False
            if words_per_subtitle and words_per_subtitle > 0:
                if len(current_subtitle) >= words_per_subtitle:
                    should_break = True
            else:
                if (len(current_text) >= chars_per_line and 
                    (current_text[-1] in ".!?,:;" or i == len(words) - 1 or (i < len(words) - 1 and words[i+1].get('type') == 'spacing'))):
                    should_break = True

            if should_break:
                # Output current subtitle
                if current_subtitle and current_start is not None and current_end is not None:
                    start_ms = int(current_start * 1000)
                    end_ms = int(current_end * 1000)
                    
                    # Split text into lines based on character limit
                    text_lines = textwrap.wrap(current_text, width=chars_per_line, break_long_words=False)
                    if not text_lines:
                        text_lines = [current_text]  # Fallback
                    
                    file_obj.write(f"{counter}\n")
                    file_obj.write(f"{format_time_ms(start_ms, start_hour)} --> {format_time_ms(end_ms, start_hour)}\n")
                    file_obj.write("\n".join(text_lines) + "\n\n")
                    counter += 1
                
                # Reset for next subtitle
                current_subtitle = []
                current_text = ""
                current_start = None
                current_end = None
        
        # Write final subtitle if there's anything left
        if current_subtitle and current_start is not None and current_end is not None:
            start_ms = int(current_start * 1000)
            end_ms = int(current_end * 1000)
            
            # Split text into lines based on character limit
            text_lines = textwrap.wrap(current_text, width=chars_per_line, break_long_words=False)
            if not text_lines:
                text_lines = [current_text]  # Fallback
            
            file_obj.write(f"{counter}\n")
            file_obj.write(f"{format_time_ms(start_ms, start_hour)} --> {format_time_ms(end_ms, start_hour)}\n")
            file_obj.write("\n".join(text_lines) + "\n\n")


def create_word_level_srt(words: List[Dict[str, Any]], output_file: Union[str, Path], 
                         remove_fillers: bool = False, filler_words: Optional[List[str]] = None,
                         fps: Optional[float] = None, fps_offset_start: int = -1, 
                         fps_offset_end: int = 0, padding_start: int = 0, padding_end: int = 0, start_hour: int = 0) -> None:
    """Word-level SRT with each word as a separate subtitle"""
    if filler_words is None:
        # Default filler words in multiple languages
        filler_words = ["uh", "um", "ah", "er", "hmm", "äh", "ähm", "hmm", "hm", "eh"]
        
    output_file = Path(output_file)
    counter = 1
    with open(output_file, 'w', encoding='utf-8') as f:
        for word in words:
            if word.get('type') == 'spacing':
                continue
            word_text = word.get('word', word.get('text', ''))
            if not word_text:
                continue
            start_time = word.get('start', 0)
            end_time = word.get('end', start_time + 0.5)
            f.write(f"{counter}\n")
            f.write(f"{format_time(start_time, start_hour)} --> {format_time(end_time, start_hour)}\n")
            f.write(f"{word_text}\n\n")
            counter += 1

def create_davinci_srt(words: List[Dict[str, Any]], output_file: Union[str, Path], 
                      silentportions: int = 0, padding_start: int = 0, padding_end: int = 0,
                      fps: Optional[float] = None, fps_offset_start: int = -1, 
                      fps_offset_end: int = 0, remove_fillers: bool = True,
                      filler_words: Optional[List[str]] = None, start_hour: int = 0) -> None:
    """Create SRT file optimized for Davinci Resolve Studio.

    Writes contiguous word sequences as blocks and places audio events (including fillers when flagged) on their own lines.
    """
    output_file = Path(output_file)
    counter = 1
    with open(output_file, 'w', encoding='utf-8') as f:
        block_words: List[Dict[str, Any]] = []
        block_start: Optional[float] = None
        block_end: Optional[float] = None

        def flush_block():
            nonlocal counter, block_words, block_start, block_end
            # Build a single large subtitle from all non-audio_event words in the block
            regular_words = [w for w in block_words if w and w.get('type') != 'audio_event']
            if not regular_words:
                block_words = []
                block_start = None
                block_end = None
                return
            start_time = block_start if block_start is not None else regular_words[0].get('start', 0)
            end_time = block_end if block_end is not None else regular_words[-1].get('end', start_time + 0.5)
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            text = ""
            for word in regular_words:
                if word.get('type') == 'word':
                    text = join_text_with_proper_spacing(text, word.get('text', ''))
            f.write(f"{counter}\n")
            f.write(f"{format_time_ms(start_ms, start_hour)} --> {format_time_ms(end_ms, start_hour)}\n")
            f.write(f"{text}\n\n")
            counter += 1
            block_words = []
            block_start = None
            block_end = None

        for w in words:
            if not w:
                continue
            wtype = w.get('type')
            if wtype == 'spacing':
                # Only break on a true pause marker meeting threshold
                is_pause = '(...)' in (w.get('text') or '')
                duration = (w.get('end', 0) - w.get('start', 0))
                duration_ms = int(round(duration * 1000)) if isinstance(duration, float) else int(duration)
                if is_pause and silentportions and duration_ms >= silentportions:
                    # Close current text block (trim end to pause start to avoid overlaps)
                    if block_words:
                        if block_end is not None and w.get('start') is not None and block_end > w['start']:
                            block_end = w['start']
                        flush_block()
                    # Write pause as its own subtitle, ensure start is not before previous block end
                    pause_start = w.get('start', 0)
                    pause_end = w.get('end', pause_start + 0.5)
                    if block_end is not None and pause_start < block_end:
                        pause_start = block_end
                    start_ms = int(pause_start * 1000)
                    end_ms = int(pause_end * 1000)
                    f.write(f"{counter}\n")
                    f.write(f"{format_time_ms(start_ms, start_hour)} --> {format_time_ms(end_ms, start_hour)}\n")
                    f.write("(...)\n\n")
                    counter += 1
                # Ignore non-pause spacing entirely (do not break blocks)
                continue

            if wtype == 'audio_event':
                # Filler or other audio event splits blocks, audio event goes on its own line
                if block_words:
                    flush_block()
                start_ms = int(w.get('start', 0) * 1000)
                end_ms = int(w.get('end', w.get('start', 0) + 0.5) * 1000)
                f.write(f"{counter}\n")
                f.write(f"{format_time_ms(start_ms, start_hour)} --> {format_time_ms(end_ms, start_hour)}\n")
                if w.get('is_filler'):
                    f.write(f"{w.get('text', '').upper()}\n\n")
                else:
                    f.write(f"({w.get('text', '')})\n\n")
                counter += 1
                continue

            # word: accumulate into large block
            if block_start is None and 'start' in w:
                block_start = w.get('start', 0)
            if 'end' in w:
                block_end = w.get('end', block_start or 0)
            block_words.append(w)

        # flush remaining text
        if block_words:
            flush_block()


def create_text_file(words: List[Dict[str, Any]], output_file: Union[str, Path]) -> None:
    """
    Create text file from words data.
    
    Args:
        words: List of word dictionaries with text info
        output_file: Path to output text file
    """
    logger.info(f"Creating text file: {output_file}")
    text = ""
    for idx, word in enumerate(words):
        word_text = word.get('word', word.get('text', ''))
        if not word_text:
            continue
        if word.get('type') == 'spacing':
            text += word_text
        else:
            prev_is_spacing = idx > 0 and words[idx-1].get('type') == 'spacing'
            if not prev_is_spacing:
                text = join_text_with_proper_spacing(text, word_text)
            else:
                text += word_text
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)
    logger.info("Text file created successfully")


def convert_to_srt(result: Dict[str, Any], output_path: Union[str, Path], 
                   fps: Optional[float] = None, fps_offset_start: int = -1, 
                   fps_offset_end: int = 0) -> None:
    """
    Convert Groq's verbose JSON output to SRT format with metadata-based filtering.
    
    Args:
        result: Transcription result dictionary from Groq API
        output_path: Path to save the SRT file
        fps: Frames per second for frame-based timing
        fps_offset_start: Frames to offset from start time (default -1)
        fps_offset_end: Frames to offset from end time (default 0)
        
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
                current_segment['text'] = join_text_with_proper_spacing(current_segment['text'], next_segment['text'])
            else:
                merged_segments.append(current_segment)
                current_segment = next_segment.copy()
        
        merged_segments.append(current_segment)

    # Apply FPS-based retiming if specified
    if fps is not None:
        retimed_segments = []
        for segment in merged_segments:
            segment_copy = segment.copy()
            
            # Convert start time to frames, apply offset, convert back to seconds
            start_frames = int(segment['start'] * fps)
            adjusted_start_frames = start_frames + fps_offset_start
            segment_copy['start'] = adjusted_start_frames / fps
            
            # Convert end time to frames, apply offset, convert back to seconds
            end_frames = int(segment['end'] * fps)
            adjusted_end_frames = end_frames + fps_offset_end
            segment_copy['end'] = adjusted_end_frames / fps
            
            retimed_segments.append(segment_copy)
        merged_segments = retimed_segments

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
            srt_lines.append(f"{format_time_ms(int(start_time * 1000))} --> {format_time_ms(int(end_time * 1000))}")
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
                srt_lines.append(f"{format_time_ms(int(chunk_start * 1000))} --> {format_time_ms(int(chunk_end * 1000))}")
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
                     subtitle_format: str, file_name: str,
                     fps: Optional[float] = None, fps_offset_start: int = -1, 
                     fps_offset_end: int = 0) -> str:
    """
    Export subtitles using AssemblyAI API.
    
    Args:
        transcript_id: ID of the transcript
        headers: Headers for API request, including Auth
        subtitle_format: Either 'srt' or 'vtt'
        file_name: Base name for output file (without extension)
        fps: Frames per second for frame-based timing
        fps_offset_start: Frames to offset from start time (default -1)
        fps_offset_end: Frames to offset from end time (default 0)
        
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
            
        # Apply FPS-based retiming if specified
        if fps is not None and subtitle_format == 'srt':
            logger.info(f"Applying FPS retiming to {filename}")
            retime_srt_file(filename, fps=fps, fps_offset_start=fps_offset_start, 
                           fps_offset_end=fps_offset_end)
            
        return filename
    else:
        raise RuntimeError(f"Subtitle export failed: {response.text}")


def custom_export_subtitles(transcript_id: str, headers: Dict[str, str], 
                           file_name: str, show_pauses: bool = False, 
                           silentportions: int = 0, chars_per_line: int = 80,
                           padding_start: int = 0, padding_end: int = 0,
                           fps: Optional[float] = None, fps_offset_start: int = -1, 
                           fps_offset_end: int = 0, remove_fillers: bool = False,
                           filler_words: Optional[List[str]] = None) -> str:
    """
    Export subtitles using AssemblyAI API with custom formatting and standardized word format.
    
    Args:
        transcript_id: ID of the transcript
        headers: Headers for API request, including Auth
        file_name: Base name for output file (without extension)
        show_pauses: Whether to show pause indicators
        silentportions: Minimum ms threshold for pause indicators
        chars_per_line: Maximum characters per line
        padding_start: Milliseconds to offset word start times
        padding_end: Milliseconds to offset word end times
        fps: Frames per second for frame-based timing
        fps_offset_start: Frames to offset from start time
        fps_offset_end: Frames to offset from end time
        remove_fillers: Whether to remove filler words
        filler_words: List of filler words to remove
        
    Returns:
        Filename of saved subtitle file
    """
    from .text_processing import standardize_word_format, process_filler_words
    
    # Get words from the transcript
    words_url = f"https://api.assemblyai.com/v2/transcript/{transcript_id}/words"
    words_response = requests.get(words_url, headers=headers)
    
    if words_response.status_code != 200:
        # Fallback to direct API export if words endpoint fails
        logger.warning(f"Failed to get word-level data, falling back to direct SRT export")
        return export_subtitles(transcript_id, headers, "srt", file_name, fps, fps_offset_start, fps_offset_end)
    
    # Parse words data
    words_data = words_response.json()
    
    # Use standardize_word_format to convert AssemblyAI format to our standard format
    processed_words = standardize_word_format(
        words_data.get('words', []),
        'assemblyai',
        show_pauses=show_pauses,
        silence_threshold=silentportions
    )
    
    # Process filler words if needed
    if remove_fillers:
        if filler_words is None:
            filler_words = ["äh", "ähm", "uh", "um", "ah", "er", "hm", "hmm"]
        processed_words = process_filler_words(processed_words, silentportions, filler_words)
    
    # Create SRT file
    srt_file = f"{file_name}.srt"
    create_srt(
        processed_words,
        srt_file,
        chars_per_line=chars_per_line,
        silentportions=silentportions,
        fps=fps,
        fps_offset_start=fps_offset_start,
        fps_offset_end=fps_offset_end,
        padding_start=padding_start,
        padding_end=padding_end,
        remove_fillers=False,  # Already handled
        show_pauses=show_pauses
    )
    
    logger.info(f"Custom SRT export saved to {srt_file}")
    return srt_file


def retime_srt_file(input_file: Union[str, Path], output_file: Optional[Union[str, Path]] = None, 
                   fps: float = 24.0, fps_offset_start: int = -1, fps_offset_end: int = 0) -> None:
    """
    Retime an existing SRT file using frame-based timing.
    
    Args:
        input_file: Path to input SRT file
        output_file: Path to output SRT file (if None, overwrites input file)
        fps: Frames per second
        fps_offset_start: Frames to offset from start time (default -1)
        fps_offset_end: Frames to offset from end time (default 0)
    
    Returns:
        None
    """
    logger.info(f"Retiming SRT file with FPS={fps}, offset_start={fps_offset_start}, offset_end={fps_offset_end}")
    
    # If no output file specified, overwrite input file
    if output_file is None:
        output_file = input_file
    
    input_file = Path(input_file)
    output_file = Path(output_file)
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input SRT file not found: {input_file}")
    
    # RegEx for SRT timestamp format: 00:00:00,000 --> 00:00:00,000
    timestamp_pattern = re.compile(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})')
    
    # Read input SRT file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Process SRT file
    retimed_lines = []
    for line in lines:
        match = timestamp_pattern.search(line)
        if match:
            start_time_str, end_time_str = match.groups()
            
            # Parse timestamps (HH:MM:SS,mmm)
            start_hours, start_minutes, start_seconds = map(float, start_time_str.replace(',', '.').split(':'))
            end_hours, end_minutes, end_seconds = map(float, end_time_str.replace(',', '.').split(':'))
            
            # Convert to seconds
            start_seconds_total = start_hours * 3600 + start_minutes * 60 + start_seconds
            end_seconds_total = end_hours * 3600 + end_minutes * 60 + end_seconds
            
            # Convert to frames, apply offset, convert back to seconds
            start_frames = int(start_seconds_total * fps)
            adjusted_start_frames = start_frames + fps_offset_start
            adjusted_start_seconds = adjusted_start_frames / fps
            
            end_frames = int(end_seconds_total * fps)
            adjusted_end_frames = end_frames + fps_offset_end
            adjusted_end_seconds = adjusted_end_frames / fps
            
            # Format back to SRT timestamp format
            retimed_line = f"{format_time(adjusted_start_seconds)} --> {format_time(adjusted_end_seconds)}"
            retimed_lines.append(line.replace(match.group(0), retimed_line))
        else:
            retimed_lines.append(line)
    
    # Write output SRT file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(retimed_lines)
    
    logger.info(f"Retimed SRT file saved to: {output_file}")
