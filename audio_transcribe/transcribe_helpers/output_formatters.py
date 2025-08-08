"""
Output formatting functions for transcription results
"""
import os
import re
import json
import requests
import textwrap
from datetime import timedelta, datetime
from pathlib import Path
from typing import Union, List, Dict, Any, Optional

# Try to import loguru, fallback to our mock implementation
try:
    from loguru import logger
except ImportError:
    import sys
    import os
    
    # Add the parent directory to sys.path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Import our mock logger
    from audio_transcribe.loguru_patch import logger


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


def format_time(seconds: float, fps: Optional[float] = None, frames_display: bool = False) -> str:
    """
    Format time in seconds to SRT timestamp format.
    
    Args:
        seconds: Time in seconds
        fps: Frames per second for frame-based timing
        frames_display: Whether to display frames in output (only applies if fps is provided)
        
    Returns:
        Formatted string in HH:MM:SS,mmm format or HH:MM:SS:FF format (if frames_display=True)
    
    From: elevenlabs - Format seconds for SRT timestamp
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_int = int(seconds % 60)
    
    if fps is not None and frames_display:
        # Calculate frames instead of milliseconds
        frames = int((seconds % 1) * fps)
        return f"{hours:02d}:{minutes:02d}:{seconds_int:02d}:{frames:02d}"
    else:
        # Standard SRT format with milliseconds
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds_int:02d},{milliseconds:03d}"


def apply_intelligent_padding(words: List[Dict[str, Any]], padding_start: int, padding_end: int) -> None:
    """
    Apply padding to word timings while respecting spacing gaps.
    
    Args:
        words: List of word dictionaries to modify in-place
        padding_start: Milliseconds to offset word start times (negative=earlier, positive=later)
        padding_end: Milliseconds to offset word end times (negative=earlier, positive=later)
    """
    if not words or (padding_start == 0 and padding_end == 0):
        return
        
    # Convert padding from ms to seconds
    padding_start_sec = padding_start / 1000.0
    padding_end_sec = padding_end / 1000.0
    
    logger.debug(f"Applying intelligent padding: start={padding_start}ms, end={padding_end}ms")
    
    for i, word in enumerate(words):
        if not word or word.get('type') == 'spacing':
            continue
            
        if 'start' not in word or 'end' not in word:
            continue
            
        # Apply start padding, respecting previous word's end time
        if padding_start != 0:
            new_start = word['start'] + padding_start_sec
            
            # Check if this would overlap with previous word
            if i > 0 and 'end' in words[i-1] and words[i-1].get('type') != 'spacing':
                prev_end = words[i-1]['end']
                # Ensure start time doesn't go before previous word's end time
                new_start = max(new_start, prev_end)
                
            # Ensure start time doesn't go after end time
            new_start = min(new_start, word['end'])
            word['start'] = new_start
            
        # Apply end padding, respecting next word's start time
        if padding_end != 0:
            new_end = word['end'] + padding_end_sec
            
            # Check if this would overlap with next word
            if i < len(words) - 1 and 'start' in words[i+1] and words[i+1].get('type') != 'spacing':
                next_start = words[i+1]['start']
                # Ensure end time doesn't go after next word's start time
                new_end = min(new_end, next_start)
                
            # Ensure end time doesn't go before start time
            new_end = max(new_end, word['start'])
            word['end'] = new_end


def process_standard_block(file, counter: int, words: List[Dict[str, Any]], 
                           start_time: float, end_time: float, chars_per_line: int) -> None:
    """
    Process and write a standard SRT block to file.
    
    Args:
        file: Open file object to write to
        counter: Current subtitle counter
        words: List of words in this block
        start_time: Start time of the block in seconds
        end_time: End time of the block in seconds
        chars_per_line: Maximum characters per line
    """
    # Build text from all words with proper spacing after punctuation
    text = ""
    for word in words:
        word_text = word.get('word', word.get('text', ''))
        text = join_text_with_proper_spacing(text, word_text)
        
    # Format into lines respecting chars_per_line
    lines = textwrap.wrap(text, width=chars_per_line)
    
    # Write to file
    file.write(f"{counter}\n")
    file.write(f"{format_time(start_time)} --> {format_time(end_time)}\n")
    file.write("\n".join(lines) + "\n\n")


def process_davinci_block(file, counter: int, words: List[Dict[str, Any]], 
                          start_time: float, end_time: float) -> None:
    """
    Process and write a DaVinci SRT block to file.
    
    Args:
        file: Open file object to write to
        counter: Current subtitle counter
        words: List of words in this block
        start_time: Start time of the block in seconds
        end_time: End time of the block in seconds
    """
    # Build text from all words with proper spacing after punctuation
    text = ""
    for word in words:
        word_text = word.get('word', word.get('text', ''))
        text = join_text_with_proper_spacing(text, word_text)
        
    # Write to file
    file.write(f"{counter}\n")
    file.write(f"{format_time(start_time)} --> {format_time(end_time)}\n")
    file.write(text.strip() + "\n\n")


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
               words_per_subtitle: int = 0) -> None:
    """
    Create SRT file from words data.
    
    Args:
        words: List of word dictionaries with timing info
        output_file: Path to output SRT file
        chars_per_line: Maximum characters per line (standard mode)
        silentportions: Minimum duration in ms to mark silent portions
        fps: Frames per second for frame-based timing
        fps_offset_start: Frames to offset from start time 
        fps_offset_end: Frames to offset from end time
        padding_start: Milliseconds to offset word start times (negative=earlier, positive=later)
        padding_end: Milliseconds to offset word end times (negative=earlier, positive=later)
        srt_mode: "standard", "word" or "davinci"
        max_words_per_block: Maximum words per subtitle block (0 = unlimited)
        remove_fillers: Remove filler words and treat them as pauses
        filler_words: List of filler words to remove (None = use defaults)
    """
    # Import locally to avoid circular imports
    from audio_transcribe.transcribe_helpers.text_processing import process_filler_words, merge_consecutive_pauses
    
    logger.info(f"Creating SRT file: {output_file} (mode: {srt_mode})")
    
    # Create a working copy of the words list
    words = [word.copy() if word else None for word in words]
    
    # Apply FPS-based retiming if specified
    if fps is not None:
        words = retime_subtitles_fps(words, fps, fps_offset_start, fps_offset_end)
    
    # Handle default settings for davinci mode
    if srt_mode == "davinci":
        # Default silent portion detection is 250ms for davinci mode if not specified
        if silentportions == 0:
            silentportions = 250
        
        # Default chars per line for davinci is 500
        chars_per_line = 500
        
        # Default max words per block for davinci is 500
        max_words_per_block = max_words_per_block or 500
        
        # Default padding start for davinci is -125ms if not specified
        if padding_start == 0:
            padding_start = -125
    
    # Apply padding but only into spacing gaps
    if padding_start != 0 or padding_end != 0:
        apply_intelligent_padding(words, padding_start, padding_end)
    
    # Process filler words if requested
    if remove_fillers:
        words = process_filler_words(words, silentportions, filler_words)
        
    with open(output_file, 'w', encoding='utf-8') as f:
        counter = 1
        
        # For word-level SRT, output each word as its own subtitle
        if srt_mode == "word":
            for word in words:
                if word.get('type') == 'spacing':
                    continue
                
                word_text = word.get('word', word.get('text', ''))
                if not word_text:
                    continue
                
                start_time = word.get('start', 0)
                end_time = word.get('end', start_time + 0.5)
                
                f.write(f"{counter}\n")
                f.write(f"{format_time(start_time)} --> {format_time(end_time)}\n")
                f.write(f"{word_text}\n\n")
                counter += 1
        else:
            # Standard or Davinci mode
            block_words = []
            block_start = None
            block_end = None
            block_text = ""
            
            for word in words:
                # Skip None or empty entries
                if not word:
                    continue
                
                # Handle spacing elements - skip but don't break block
                if word.get('type') == 'spacing':
                    # Add spacing text to block_text
                    block_text += word.get('text', '')
                    continue
                
                # Get word text and timing
                word_text = word.get('word', word.get('text', ''))
                if not word_text:
                    continue
                
                start_time = word.get('start', 0)
                end_time = word.get('end', start_time + 0.5)
                
                # Initialize block timing if this is the first word
                if block_start is None:
                    block_start = start_time
                    block_end = end_time
                else:
                    # Update block end time
                    block_end = end_time
                
                # Add word to block
                block_words.append(word)
                block_text += word_text
                
                # Check if we should output this block
                if srt_mode == "davinci":
                    # For davinci mode, we add a space after each word
                    # and check if we've reached max words per block
                    block_text += " "
                    
                    # Output block if we've reached max words
                    if len(block_words) >= max_words_per_block:
                        process_davinci_block(f, counter, block_words, block_start, block_end)
                        counter += 1
                        
                        # Reset block
                        block_words = []
                        block_text = ""
                        block_start = None
                        block_end = None
                else:
                    # For standard mode, we check if adding this word exceeds chars_per_line
                    # If so, we output the current block and start a new one
                    
                    # First, check if current word alone exceeds chars_per_line
                    if len(word_text) > chars_per_line:
                        # Word itself is too long, needs to be in its own block
                        # First output any pending block if exists
                        if len(block_words) > 1:
                            # Get timing from all words except current one
                            prev_words = block_words[:-1]
                            prev_start = prev_words[0].get('start', 0)
                            prev_end = prev_words[-1].get('end', 0)
                            
                            # Output previous words as a block
                            process_standard_block(f, counter, prev_words, prev_start, prev_end, chars_per_line)
                            counter += 1
                            
                        # Output current word as its own block
                        process_standard_block(f, counter, [word], start_time, end_time, chars_per_line)
                        counter += 1
                        
                        # Reset block
                        block_words = []
                        block_text = ""
                        block_start = None
                        block_end = None
                    elif (words_per_subtitle and len(block_words) >= words_per_subtitle) or len(block_text) > chars_per_line:
                        # Current block exceeds chars_per_line with this word
                        # Output all but current word
                        prev_words = block_words[:-1]
                        if prev_words:
                            prev_start = prev_words[0].get('start', 0)
                            prev_end = prev_words[-1].get('end', 0)
                            
                            # Output previous words as a block
                            process_standard_block(f, counter, prev_words, prev_start, prev_end, chars_per_line)
                            counter += 1
                            
                            # Start new block with current word
                            block_words = [word]
                            block_text = word_text
                            block_start = start_time
                            block_end = end_time
                        else:
                            # No previous words, current word is too long
                            process_standard_block(f, counter, [word], start_time, end_time, chars_per_line)
                            counter += 1
                            
                            # Reset block
                            block_words = []
                            block_text = ""
                            block_start = None
                            block_end = None
            
            # Output any remaining words
            if block_words:
                if srt_mode == "davinci":
                    process_davinci_block(f, counter, block_words, block_start, block_end)
                else:
                    process_standard_block(f, counter, block_words, block_start, block_end, chars_per_line)
    
    # For SRT files with silent portions or padding, merge consecutive pauses
    if silentportions > 0 or padding_start != 0 or padding_end != 0:
        merge_consecutive_pauses(output_file)


def create_standard_srt(words: List[Dict[str, Any]], output_file: Union[str, Path], 
                       chars_per_line: int = 80, silentportions: int = 0,
                       fps: Optional[float] = None, fps_offset_start: int = -1, 
                       fps_offset_end: int = 0, padding_start: int = 0, padding_end: int = 0) -> None:
    """Standard SRT format with character limits per line"""
    create_srt(
        words=words,
        output_file=output_file,
        chars_per_line=chars_per_line,
        silentportions=silentportions,
        fps=fps,
        fps_offset_start=fps_offset_start,
        fps_offset_end=fps_offset_end,
        padding_start=padding_start,
        padding_end=padding_end,
        srt_mode="standard"
    )

def create_word_level_srt(words: List[Dict[str, Any]], output_file: Union[str, Path], 
                         remove_fillers: bool = False, filler_words: Optional[List[str]] = None,
                         fps: Optional[float] = None, fps_offset_start: int = -1, 
                         fps_offset_end: int = 0, padding_start: int = 0, padding_end: int = 0) -> None:
    """Word-level SRT with each word as a separate subtitle"""
    create_srt(
        words=words,
        output_file=output_file,
        fps=fps,
        fps_offset_start=fps_offset_start,
        fps_offset_end=fps_offset_end,
        padding_start=padding_start,
        padding_end=padding_end,
        srt_mode="word",
        remove_fillers=remove_fillers,
        filler_words=filler_words
    )

def create_davinci_srt(words: List[Dict[str, Any]], output_file: Union[str, Path], 
                      silentportions: int = 0, padding_start: int = 0, padding_end: int = 0,
                      fps: Optional[float] = None, fps_offset_start: int = -1, 
                      fps_offset_end: int = 0, remove_fillers: bool = True,
                      filler_words: Optional[List[str]] = None) -> None:
    """Create SRT file optimized for Davinci Resolve Studio"""
    create_srt(
        words=words,
        output_file=output_file,
        silentportions=silentportions,
        padding_start=padding_start,
        padding_end=padding_end,
        fps=fps,
        fps_offset_start=fps_offset_start,
        fps_offset_end=fps_offset_end,
        srt_mode="davinci",
        max_words_per_block=500,
        remove_fillers=remove_fillers,
        filler_words=filler_words
    )


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
            transcript = join_text_with_proper_spacing(transcript, text)
    
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
    from audio_transcribe.transcribe_helpers.text_processing import standardize_word_format, process_filler_words
    
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
        remove_fillers=False  # Already handled
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
