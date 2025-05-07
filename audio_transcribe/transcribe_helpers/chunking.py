"""
Audio chunking functions for transcription
"""
import time
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple, Optional
from pydub import AudioSegment
from loguru import logger
from .text_processing import find_longest_common_sequence


def split_audio(audio_path: Union[str, Path], chunk_length: int = 600, overlap: int = 10) -> List[Tuple[Path, int]]:
    """
    Split audio file into chunks with overlap.
    
    Args:
        audio_path: Path to audio file
        chunk_length: Length of each chunk in seconds
        overlap: Overlap between chunks in seconds
        
    Returns:
        List of (chunk_path, start_time_ms) tuples
    
    From: groq - Split audio into processable chunks
    """
    audio_path = Path(audio_path)
    audio = AudioSegment.from_file(audio_path)
    duration = len(audio)
    
    logger.info(f"Audio duration: {duration/1000:.2f}s")
    
    # Calculate # of chunks
    chunk_ms = chunk_length * 1000
    overlap_ms = overlap * 1000
    total_chunks = (duration // (chunk_ms - overlap_ms)) + 1
    
    logger.info(f"Processing {total_chunks} chunks...")
    
    chunks = []
    
    # Loop through each chunk, extract current chunk from audio
    for i in range(total_chunks):
        start = i * (chunk_ms - overlap_ms)
        end = min(start + chunk_ms, duration)
        
        logger.info(f"Processing chunk {i+1}/{total_chunks}")
        logger.info(f"Time range: {start/1000:.1f}s - {end/1000:.1f}s")
        
        chunk = audio[start:end]
        
        # Save chunk to temporary file
        with tempfile.NamedTemporaryFile(suffix='.flac', delete=False) as temp_file:
            chunk_path = Path(temp_file.name)
            chunk.export(chunk_path, format='flac')
            chunks.append((chunk_path, start))
    
    return chunks


def merge_transcripts(results: List[Tuple[Dict[str, Any], int]], overlap: int = 10) -> Dict[str, Any]:
    """
    Merge transcription chunks and handle overlaps.
    
    Args:
        results: List of (result, start_time_ms) tuples
        overlap: Overlap between chunks in seconds
        
    Returns:
        Merged transcription dictionary
    
    From: groq - Combine transcript chunks with overlaps
    """
    logger.info("\nMerging results...")
    final_segments = []
    
    # Process each chunk's segments and adjust timestamps
    processed_chunks = []
    overlap_sec = overlap  # Convert overlap to seconds
    
    for i, (chunk, chunk_start_ms) in enumerate(results):
        # Extract full segment data including metadata
        data = chunk.model_dump() if hasattr(chunk, 'model_dump') else chunk
        segments = data['segments']
        
        # Convert chunk_start_ms from milliseconds to seconds for timestamp adjustment
        chunk_start_sec = chunk_start_ms / 1000.0
        
        # Adjust all timestamps in this chunk based on its position in the original audio
        adjusted_segments = []
        for segment in segments:
            adjusted_segment = segment.copy()
            # Adjust start and end times based on the chunk's position
            # For chunks after the first one, subtract the overlap time
            if i > 0:
                adjusted_segment['start'] += chunk_start_sec - (i * overlap_sec)
                adjusted_segment['end'] += chunk_start_sec - (i * overlap_sec)
            else:
                adjusted_segment['start'] += chunk_start_sec
                adjusted_segment['end'] += chunk_start_sec
            adjusted_segments.append(adjusted_segment)
        
        # If not last chunk, find next chunk start time
        if i < len(results) - 1:
            next_start = results[i + 1][1]  # in milliseconds
            
            # Split segments into current and overlap based on next chunk's start time
            current_segments = []
            overlap_segments = []
            
            for segment in adjusted_segments:
                if segment['end'] * 1000 > next_start:
                    overlap_segments.append(segment)
                else:
                    current_segments.append(segment)
            
            # Merge overlap segments if any exist
            if overlap_segments:
                merged_overlap = overlap_segments[0].copy()
                merged_overlap.update({
                    'text': ' '.join(s['text'] for s in overlap_segments),
                    'end': overlap_segments[-1]['end']
                })
                current_segments.append(merged_overlap)
                
            processed_chunks.append(current_segments)
        else:
            # For last chunk, keep all segments
            processed_chunks.append(adjusted_segments)
    
    # Merge boundaries between chunks
    for i in range(len(processed_chunks) - 1):
        # Add all segments except last from current chunk
        final_segments.extend(processed_chunks[i][:-1])
        
        # Merge boundary segments
        last_segment = processed_chunks[i][-1]
        first_segment = processed_chunks[i + 1][0]
        
        merged_text = find_longest_common_sequence([last_segment['text'], first_segment['text']])
        merged_segment = last_segment.copy()
        merged_segment.update({
            'text': merged_text,
            'end': first_segment['end']
        })
        final_segments.append(merged_segment)
    
    # Add all segments from last chunk
    if processed_chunks:
        final_segments.extend(processed_chunks[-1])
    
    # Create final transcription
    final_text = ' '.join(segment['text'] for segment in final_segments)
    
    return {
        "text": final_text,
        "segments": final_segments
    }


def transcribe_with_chunks(audio_path: Union[str, Path], 
                          transcribe_function: callable, 
                          chunk_length: int = 600, 
                          overlap: int = 10) -> Dict[str, Any]:
    """
    Split audio and process with a transcribe function, then merge results.
    
    Args:
        audio_path: Path to the audio file
        transcribe_function: Function that takes a file path and returns a transcription result
        chunk_length: Length of each chunk in seconds
        overlap: Overlap between chunks in seconds
        
    Returns:
        Merged transcription dictionary
        
    From: new - Generic chunking and transcription
    """
    audio_path = Path(audio_path)
    logger.info(f"Starting chunked transcription of: {audio_path}")
    
    # Get chunks with their start times
    chunks = split_audio(audio_path, chunk_length=chunk_length, overlap=overlap)
    
    results = []
    
    # Process each chunk with the provided transcribe function
    for i, (chunk_path, start_time) in enumerate(chunks):
        logger.info(f"Transcribing chunk {i+1}/{len(chunks)}")
        
        try:
            # Call the transcribe function on this chunk
            result = transcribe_function(chunk_path)
            results.append((result, start_time))
        except Exception as e:
            logger.error(f"Error transcribing chunk {i+1}: {str(e)}")
            raise
        finally:
            # Clean up the temporary chunk file
            chunk_path.unlink(missing_ok=True)
    
    # Merge the transcripts
    merged_result = merge_transcripts(results, overlap)
    
    return merged_result
