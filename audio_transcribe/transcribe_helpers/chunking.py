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


def split_audio(audio_path: Union[str, Path], chunk_length: int = 600, overlap: int = 10) -> List[Tuple[Path, float]]:
    """
    Split audio file into chunks with overlap.
    
    Args:
        audio_path: Path to audio file
        chunk_length: Length of each chunk in seconds
        overlap: Overlap between chunks in seconds
        
    Returns:
        List of (chunk_path, start_time_seconds) tuples
    
    From: groq - Split audio into processable chunks
    """
    audio_path = Path(audio_path)
    audio = AudioSegment.from_file(audio_path)
    duration_ms = len(audio)
    duration_seconds = duration_ms / 1000.0
    
    logger.info(f"Audio duration: {duration_seconds:.2f}s")
    
    # Calculate # of chunks
    chunk_seconds = chunk_length
    overlap_seconds = overlap
    effective_chunk_length = chunk_seconds - overlap_seconds
    total_chunks = int((duration_seconds / effective_chunk_length)) + 1 if effective_chunk_length > 0 else 1
    
    logger.info(f"Processing {total_chunks} chunks...")
    
    chunks = []
    
    # Loop through each chunk, extract current chunk from audio
    for i in range(total_chunks):
        start_seconds = i * effective_chunk_length
        end_seconds = min(start_seconds + chunk_seconds, duration_seconds)
        
        # Convert to milliseconds for AudioSegment slicing
        start_ms = int(start_seconds * 1000)
        end_ms = int(end_seconds * 1000)
        
        logger.info(f"Processing chunk {i+1}/{total_chunks}")
        logger.info(f"Time range: {start_seconds:.1f}s - {end_seconds:.1f}s")
        
        chunk = audio[start_ms:end_ms]
        
        # Save chunk to temporary file
        with tempfile.NamedTemporaryFile(suffix='.flac', delete=False) as temp_file:
            chunk_path = Path(temp_file.name)
            chunk.export(chunk_path, format='flac')
            chunks.append((chunk_path, start_seconds))
    
    return chunks


def merge_transcripts(results: List[Tuple[Dict[str, Any], float]], overlap: int = 10) -> Dict[str, Any]:
    """
    Merge transcription chunks and handle overlaps.
    
    Args:
        results: List of (result, start_time_seconds) tuples
        overlap: Overlap between chunks in seconds
        
    Returns:
        Merged transcription dictionary
    
    From: groq - Combine transcript chunks with overlaps
    """
    logger.info("\nMerging results...")
    final_segments = []
    
    # Process each chunk's segments and adjust timestamps
    processed_chunks = []
    overlap_sec = overlap
    
    for i, (chunk, chunk_start_sec) in enumerate(results):
        # Extract full segment data including metadata
        data = chunk.model_dump() if hasattr(chunk, 'model_dump') else chunk
        segments = data['segments']
        
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
            next_start_sec = results[i + 1][1]  # in seconds
            
            # Split segments into current and overlap based on next chunk's start time
            current_segments = []
            overlap_segments = []
            
            for segment in adjusted_segments:
                if segment['end'] > next_start_sec:
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
    Transcribe an audio file by splitting it into chunks with overlap.
    
    Args:
        audio_path: Path to the audio file
        transcribe_function: Function to call for transcribing each chunk
        chunk_length: Length of each chunk in seconds
        overlap: Overlap between chunks in seconds
        
    Returns:
        Dictionary with merged transcription results
    """
    from tempfile import NamedTemporaryFile
    import os
    
    # Import here to avoid circular imports
    from .audio_processing import preprocess_audio
    from pydub import AudioSegment
    
    logger.info(f"Transcribing {audio_path} in chunks")
    
    # Step 1: Preprocess audio if needed
    processed_path = preprocess_audio(audio_path)
    
    # Step 2: Split audio into chunks
    chunks = split_audio(processed_path, chunk_length, overlap)
    logger.info(f"Split audio into {len(chunks)} chunks")
    
    # Lists to store results and track failed chunks
    results = []
    failed_chunks = []
    temp_chunks = []  # Store all generated temp chunks for cleanup in case of error
    
    # Process each chunk with the provided transcribe function
    for i, (chunk_path, start_time_seconds) in enumerate(chunks):
        logger.info(f"Transcribing chunk {i+1}/{len(chunks)}")
        temp_chunks.append(chunk_path)  # Track for cleanup
        
        try:
            # Call the transcribe function on this chunk
            # The transcribe_function is a lambda that calls api_instance.transcribe with the correct API
            result = transcribe_function(chunk_path)
            
            # Check if result is valid
            if not result or (isinstance(result, dict) and not result.get('text') and not result.get('segments')):
                logger.warning(f"Chunk {i+1} returned empty or invalid result")
                failed_chunks.append((i+1, start_time_seconds, "Empty or invalid result"))
            else:
                results.append((result, start_time_seconds))
                
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error transcribing chunk {i+1}: {error_message}")
            failed_chunks.append((i+1, start_time, error_message))
            
            # Immediately stop processing if rate limit error is detected
            if "429" in error_message or "rate limit" in error_message.lower() or "too many requests" in error_message.lower():
                logger.error("Rate limit exceeded. Stopping all processing immediately.")
                
                # Clean up ALL chunk files before exiting
                for chunk_file in temp_chunks:
                    try:
                        if chunk_file.exists():
                            chunk_file.unlink()
                            logger.debug(f"Removed temporary chunk file: {chunk_file}")
                    except Exception as cleanup_err:
                        logger.debug(f"Failed to clean up chunk file {chunk_file}: {cleanup_err}")
                
                # Extract retry information if available
                import re
                retry_after = None
                retry_patterns = [
                    r"try again in (\d+[ms][\d\.]*)",
                    r"retry after (\d+[ms][\d\.]*)",
                    r"retry in (\d+[ms][\d\.]*)",
                    r"available in (\d+[ms][\d\.]*)"
                ]
                
                for pattern in retry_patterns:
                    match = re.search(pattern, error_message, re.IGNORECASE)
                    if match:
                        retry_after = match.group(1)
                        break
                
                retry_msg = f" Please try again after {retry_after}." if retry_after else ""
                raise RuntimeError(f"Rate limit exceeded.{retry_msg} Stopping all processing immediately.")
        finally:
            # Clean up the temporary chunk file
            try:
                if chunk_path.exists():
                    chunk_path.unlink()
                    temp_chunks.remove(chunk_path)  # Remove from cleanup list
            except Exception as cleanup_err:
                logger.debug(f"Failed to clean up chunk file in main loop: {cleanup_err}")
    
    # If all chunks failed, propagate the error
    if not results and failed_chunks:
        first_error = failed_chunks[0][2]
        raise Exception(f"All chunks failed transcription. First error: {first_error}")
    
    # Merge the transcripts from successful chunks
    merged_result = merge_transcripts(results, overlap)
    
    # Add information about failed chunks to the result
    merged_result["chunk_status"] = {
        "total_chunks": len(chunks),
        "successful_chunks": len(results),
        "failed_chunks": failed_chunks
    }
    
    # Check if we have enough successful chunks for a good transcription
    if len(results) < len(chunks) * 0.5:  # Less than 50% successful
        logger.warning(f"Only {len(results)}/{len(chunks)} chunks were successfully transcribed")
        merged_result["warning"] = f"Transcription may be incomplete. Only {len(results)}/{len(chunks)} chunks were successful."
    
    return merged_result
