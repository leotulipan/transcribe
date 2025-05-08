"""
Text processing functions for transcription
"""
import re
from typing import List, Dict, Any, Union, Optional
from pathlib import Path

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


def standardize_word_format(basic_words: List[Dict[str, Any]], 
                            show_pauses: bool = False, silence_threshold: int = 250) -> List[Dict[str, Any]]:
    """
    Takes a basic list of word dicts (with integer ms times) and adds 
    spacing elements between words and at the start if necessary.
    Also identifies words that are just spaces.
    
    Args:
        basic_words: List of word dictionaries from a basic parser (int ms times).
        show_pauses: Whether to add "(...)" text for significant pauses.
        silence_threshold: Minimum ms duration for a gap to be marked as a pause.
        
    Returns:
        Standardized list including word and spacing elements (int ms times).
    """
    standardized = []
    
    if not basic_words:
        return standardized
        
    logger.debug(f"Standardizing {len(basic_words)} basic words into intermediary format.")

    # Ensure input times are integers
    words = []
    for w in basic_words:
        w_copy = w.copy()
        w_copy['start'] = int(round(w_copy.get('start', 0)))
        w_copy['end'] = int(round(w_copy.get('end', 0)))
        # Mark words that are actually spaces
        if w_copy.get('text', "").strip() == "":
            w_copy['type'] = "spacing"
        else:
             w_copy['type'] = "word" # Ensure type is set
        words.append(w_copy)
        
    # Add initial spacing if the first element isn't spacing and doesn't start at 0
    if words[0]['type'] != 'spacing' and words[0]['start'] > 0:
        pause_text = " "
        if show_pauses and words[0]['start'] > silence_threshold:
            pause_text = " (...) "
            
        standardized.append({
            'text': pause_text,
            'start': 0,
            'end': words[0]['start'],
            'type': 'spacing',
            'speaker_id': words[0].get('speaker_id', '')
        })
        
    # Add words and spacings between them
    for i, word in enumerate(words):
        standardized.append(word) # Add the word itself (or space if type changed)
        
        # Add spacing element *after* this word if there is a next word
        if i < len(words) - 1:
            next_word = words[i+1]
            gap = next_word['start'] - word['end']
            
            if gap > 0:
                pause_text = " "
                if show_pauses and gap > silence_threshold:
                    pause_text = " (...) "
                
                standardized.append({
                    'text': pause_text,
                    'start': word['end'],
                    'end': next_word['start'],
                    'type': 'spacing',
                    # Inherit speaker from the preceding word for the gap
                    'speaker_id': word.get('speaker_id', '') 
                })
            elif gap < 0:
                 logger.warning(f"Overlap detected between word {i} ({word.get('text')}) and word {i+1} ({next_word.get('text')}). Start/End: {word['end']} / {next_word['start']}")

    logger.debug(f"Standardized to {len(standardized)} elements (words and spaces)")
    if len(standardized) > 1:
        # Use index 1 if it exists, otherwise index 0
        idx_to_log = 1 if len(standardized) > 1 else 0
        w = standardized[idx_to_log]
        logger.debug(f"Word/Space at index {idx_to_log}: text='{w.get('text')}', type={w.get('type')}, start={w.get('start')}, end={w.get('end')}")
        
    return standardized


def process_filler_words(words: List[Dict[str, Any]], pause_threshold: int, 
                        filler_words: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Pre-process words to remove filler words and handle them as pauses.
    Also removes audio_events and text in parentheses.
    
    Args:
        words: List of word dictionaries with timing info
        pause_threshold: Minimum duration in ms to mark silent portions
        filler_words: List of filler words to remove (if None, use defaults)
        
    Returns:
        Processed list of words with fillers removed
    """
    if not words:
        return words
    
    if filler_words is None:
        filler_words = ["äh", "ähm"]
    
    logger.info(f"Processing words to remove filler words and audio events: {filler_words}")
    
    processed_words = []
    i = 0
    
    while i < len(words):
        # Skip None or empty entries
        if not words[i]:
            i += 1
            continue
        
        # Check if current item is an audio_event
        if words[i].get('type') == 'audio_event':
            # Check if we have both previous and next spacing
            prev_spacing = i > 0 and words[i-1].get('type') == 'spacing'
            next_spacing = i < len(words) - 1 and words[i+1].get('type') == 'spacing'
            
            if prev_spacing and next_spacing:
                # Create a new spacing element replacing previous spacing + event + next spacing
                merged_pause = {
                    'type': 'spacing',
                    'start': words[i-1]['start'],
                    'end': words[i+1]['end'],
                    'text': ' ',
                    'speaker_id': words[i].get('speaker_id', '')
                }
                processed_words.append(merged_pause)
                # Skip the audio event and the next spacing
                i += 2
                continue
            # If we don't have spacings on both sides, just skip the audio event
            i += 1
            continue
        
        # Check if current item is a word with text in parentheses
        if words[i].get('type') == 'word' and words[i].get('text', '').startswith('(') and words[i].get('text', '').endswith(')'):
            # Check if we have both previous and next spacing
            prev_spacing = i > 0 and words[i-1].get('type') == 'spacing'
            next_spacing = i < len(words) - 1 and words[i+1].get('type') == 'spacing'
            
            if prev_spacing and next_spacing:
                # Create a new spacing element replacing previous spacing + parenthesized word + next spacing
                merged_pause = {
                    'type': 'spacing',
                    'start': words[i-1]['start'],
                    'end': words[i+1]['end'],
                    'text': ' ',
                    'speaker_id': words[i].get('speaker_id', '')
                }
                processed_words.append(merged_pause)
                # Skip the parenthesized word and the next spacing
                i += 2
                continue
            # If we don't have spacings on both sides, just skip the parenthesized word
            i += 1
            continue
        
        # Check if current item is a word with text that could be a filler word
        if words[i].get('type') == 'word':
            word_text = words[i].get('text', '').lower()
            
            # If the word is a filler word
            if word_text in filler_words:
                # Check if we have both previous and next spacing
                prev_spacing = i > 0 and words[i-1].get('type') == 'spacing'
                next_spacing = i < len(words) - 1 and words[i+1].get('type') == 'spacing'
                
                if prev_spacing and next_spacing:
                    # Create a new spacing element replacing previous spacing + filler + next spacing
                    filler_pause = {
                        'type': 'spacing',
                        'start': words[i-1]['start'],
                        'end': words[i+1]['end'],
                        'text': ' ',
                        'speaker_id': words[i].get('speaker_id', '')
                    }
                    
                    # Apply padding to the preceding word if it exists
                    if len(processed_words) > 0 and processed_words[-1].get('type') == 'word':
                        # Record the original end time
                        orig_end = processed_words[-1]['end']
                        # Add padding (convert to seconds)
                        padding = 30  # Default padding in ms
                        processed_words[-1]['end'] += padding / 1000.0
                        
                        logger.debug(f"Added {padding}ms padding to word ending at {orig_end}")
                    
                    # Add the filler pause
                    processed_words.append(filler_pause)
                    
                    # Skip the filler word and the next spacing
                    i += 2
                    continue
            
            # Not a filler word or not surrounded by spacings, add as-is
            processed_words.append(words[i])
        else:
            # Not a word, add as-is
            processed_words.append(words[i])
        
        i += 1
    
    logger.info(f"Word processing complete. Removed {len(words) - len(processed_words)} elements")
    return processed_words


def merge_consecutive_pauses(srt_file: Union[str, Path]) -> None:
    """
    Post-process SRT file to merge consecutive pause entries.
    
    Args:
        srt_file: Path to SRT file
    
    From: elevenlabs - Combine adjacent pause markers
    """
    import re
    from .utils import min_timestamp, max_timestamp
    
    logger.info("Post-processing SRT file to merge consecutive pauses...")
    
    # Read the SRT file
    with open(srt_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into entries
    entries = re.split(r'\n\n+', content.strip())
    
    # Process entries
    merged_entries = []
    i = 0
    while i < len(entries):
        if i < len(entries) - 1:
            current = entries[i]
            next_entry = entries[i + 1]
            
            # Check if both are pause entries
            current_is_pause = "(...)" in current
            next_is_pause = "(...)" in next_entry
            
            if current_is_pause and next_is_pause:
                # Extract timestamps from current entry
                current_match = re.search(r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n', current)
                
                # Extract timestamps from next entry
                next_match = re.search(r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n', next_entry)
                
                if current_match and next_match:
                    current_index = current_match.group(1)
                    current_start = current_match.group(2)
                    current_end = current_match.group(3)
                    
                    next_start = next_match.group(2)
                    next_end = next_match.group(3)
                    
                    # Get the earlier start time and later end time
                    merged_start = min_timestamp(current_start, next_start)
                    merged_end = max_timestamp(current_end, next_end)
                    
                    # Create merged entry
                    merged_entry = f"{current_index}\n{merged_start} --> {merged_end}\n(...)"
                    merged_entries.append(merged_entry)
                    
                    # Skip the next entry
                    i += 2
                    continue
                else:
                    # If regex matching failed, just add the current entry as-is
                    logger.warning(f"Failed to extract timestamps from pause entries")
                    merged_entries.append(current)
                    i += 1
                    continue
        
        # If no merge happened, add the current entry as is
        merged_entries.append(entries[i])
        i += 1
    
    # Write back the merged content
    with open(srt_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(merged_entries))
    
    logger.info(f"Merged {len(entries) - len(merged_entries)} consecutive pause entries")


def find_longest_common_sequence(sequences: List[str], match_by_words: bool = True) -> str:
    """
    Find the optimal alignment between sequences with longest common sequence 
    and sliding window matching.
    
    Args:
        sequences: List of text sequences to align and merge
        match_by_words: Whether to match by words or characters
        
    Returns:
        Merged sequence with optimal alignment
    
    From: groq - Find optimal text alignment
    """
    if not sequences:
        return ""

    # Convert input based on matching strategy
    if match_by_words:
        sequences = [
            [word for word in re.split(r'(\s+\w+)', seq) if word]
            for seq in sequences
        ]
    else:
        sequences = [list(seq) for seq in sequences]

    left_sequence = sequences[0]
    left_length = len(left_sequence)
    total_sequence = []

    for right_sequence in sequences[1:]:
        max_matching = 0.0
        right_length = len(right_sequence)
        max_indices = (left_length, left_length, 0, 0)

        # Try different alignments
        for i in range(1, left_length + right_length + 1):
            # Add epsilon to favor longer matches
            eps = float(i) / 10000.0

            left_start = max(0, left_length - i)
            left_stop = min(left_length, left_length + right_length - i)
            left = left_sequence[left_start:left_stop]

            right_start = max(0, i - left_length)
            right_stop = min(right_length, i)
            right = right_sequence[right_start:right_stop]

            if len(left) != len(right):
                logger.warning("Mismatched subsequences detected during transcript merging")
                continue

            matches = sum(a == b for a, b in zip(left, right))
            
            # Normalize matches by position and add epsilon 
            matching = matches / float(i) + eps

            # Require at least 2 matches
            if matches > 1 and matching > max_matching:
                max_matching = matching
                max_indices = (left_start, left_stop, right_start, right_stop)

        # Use the best alignment found
        left_start, left_stop, right_start, right_stop = max_indices
        
        # Take left half from left sequence and right half from right sequence
        left_mid = (left_stop + left_start) // 2
        right_mid = (right_stop + right_start) // 2
        
        total_sequence.extend(left_sequence[:left_mid])
        left_sequence = right_sequence[right_mid:]
        left_length = len(left_sequence)

    # Add remaining sequence
    total_sequence.extend(left_sequence)
    
    # Join back into text
    if match_by_words:
        return ''.join(total_sequence)
    return ''.join(total_sequence)


def segments_to_words(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert segment-level transcription to word-level format.
    
    Args:
        segments: List of transcript segments with text and timing info
        
    Returns:
        List of word-level entries with timing info
    
    From: new - Convert segments to word-level format
    """
    words = []
    
    for segment in segments:
        if 'text' not in segment or not segment['text'].strip():
            continue
            
        # Get segment timing info
        start_time = segment.get('start', 0)
        end_time = segment.get('end', 0)
        segment_duration = end_time - start_time
        
        # Split the text into words
        text_words = segment['text'].split()
        if not text_words:
            continue
            
        # Calculate approximate time per word
        time_per_word = segment_duration / len(text_words)
        
        # Create word entries with estimated timing
        for i, word_text in enumerate(text_words):
            word_start = start_time + (i * time_per_word)
            word_end = word_start + time_per_word
            
            word_entry = {
                'type': 'word',
                'start': word_start,
                'end': word_end,
                'text': word_text,
                'speaker_id': segment.get('speaker', segment.get('speaker_id', ''))
            }
            
            words.append(word_entry)
            
            # Add spacing between words (except after the last word)
            if i < len(text_words) - 1:
                spacing_duration = 0.01  # 10ms spacing
                spacing_entry = {
                    'type': 'spacing',
                    'start': word_end,
                    'end': word_end + spacing_duration,
                    'text': ' ',
                    'speaker_id': segment.get('speaker', segment.get('speaker_id', ''))
                }
                words.append(spacing_entry)
    
    return words
