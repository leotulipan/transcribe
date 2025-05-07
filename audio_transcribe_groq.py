#!/usr/bin/env python3
# /// script
# dependencies = [
#   "groq",
#   "pydub",
#   "pathlib",
#   "datetime",
#   "argparse",
#   "python-dotenv",
#   "librosa",
#   "loguru",
# ]
# ///

import glob
from groq import Groq, RateLimitError
from pydub import AudioSegment
import json
from pathlib import Path
from datetime import datetime
import time
import os
import tempfile
import re
import argparse
from dotenv import load_dotenv
import sys
from loguru import logger

# Import transcribe_helpers package
from transcribe_helpers import (
    # audio_processing
    check_audio_length, check_audio_format, convert_to_flac, convert_to_pcm, check_file_size,
    # utils
    setup_logger, check_transcript_exists, 
    # output_formatters
    create_srt, create_davinci_srt, create_text_file, create_word_level_srt
)

# Global variables
args = None
MAX_AUDIO_LENGTH = 7200  # seconds
MAX_DAVINCI_WORDS = 500  # maximum words per subtitle block for davinci mode
FILLER_WORDS = ["äh", "ähm"]  # List of filler words to potentially remove

def get_args():
    parser = argparse.ArgumentParser(description="Audio transcription using Groq API.")
    parser.add_argument("audio_path", type=str, help="Path to the input audio file or pattern.")
    parser.add_argument("-d", "--debug", help="Debug mode", action="store_true")
    parser.add_argument("-c", "--chars_per_line", type=int, default=80, help="Maximum characters per line in SRT file")
    parser.add_argument("-C", "--word-srt", action="store_true", help="Output SRT with each word as its own subtitle (word-level SRT)")
    parser.add_argument("-s", "--speaker_labels", help="Use this flag to remove speaker labels", action="store_false", default=True)
    parser.add_argument("--keep-flac", help="Keep the generated FLAC file after processing", action="store_true")
    parser.add_argument("--no-convert", help="Send the audio file as-is without conversion", action="store_true")
    parser.add_argument("--use-pcm", help="Use PCM format instead of FLAC (larger file size)", action="store_true")
    parser.add_argument("-l", "--language", help="Language code (ISO-639-1 or ISO-639-3)", default="en")
    parser.add_argument("-v", "--verbose", help="Show all log messages in console", action="store_true")
    parser.add_argument("--force", help="Force re-transcription even if files exist", action="store_true")
    parser.add_argument("-p", "--silentportions", type=int, help="Mark pauses longer than X milliseconds with (...)", default=0)
    parser.add_argument("--davinci-srt", "-D", help="Export SRT for Davinci Resolve with optimized subtitle blocks", action="store_true")
    parser.add_argument("--padding", type=int, help="Add X milliseconds padding to word end times (default: 30ms)", default=30)
    parser.add_argument("--remove-fillers", help="Remove filler words like 'äh' and 'ähm' and treat them as pauses", action="store_true")
    parser.add_argument("-m", "--model", type=str, default="whisper-large-v3", help="Model to use (default: whisper-large-v3)")
    parser.add_argument("--chunk-length", type=int, default=600, help="Length of each chunk in seconds (default: 600)")
    parser.add_argument("--overlap", type=int, default=10, help="Overlap between chunks in seconds (default: 10)")
    parser.add_argument("--fps", type=float, help="Frames per second for frame-based editing (e.g., 24, 29.97, 30)", default=None)
    parser.add_argument("--fps-offset-start", type=int, help="Frames to offset from start time (default: 1)", default=1)
    parser.add_argument("--fps-offset-end", type=int, help="Frames to offset from end time (default: 0)", default=0)
    return parser.parse_args()

def find_longest_common_sequence(sequences: list[str], match_by_words: bool = True) -> str:
    """
    Find the optimal alignment between sequences with longest common sequence and sliding window matching.
    
    Args:
        sequences: List of text sequences to align and merge
        match_by_words: Whether to match by words (True) or characters (False)
        
    Returns:
        str: Merged sequence with optimal alignment
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

def merge_transcripts(results: list[tuple[dict, int]], overlap: int = 10) -> dict:
    """
    Merge transcription chunks and handle overlaps.
    
    Args:
        results: List of (result, start_time) tuples
        overlap: Overlap between chunks in seconds
        
    Returns:
        dict: Merged transcription
    """
    logger.info("Merging transcription results...")
    
    has_words = False
    words = []
    
    for chunk, chunk_start_ms in results:
        # Convert Pydantic model to dict if needed
        data = chunk.model_dump() if hasattr(chunk, 'model_dump') else chunk
        
        # Process word timestamps if available
        if isinstance(data, dict) and 'words' in data and data['words'] is not None and len(data['words']) > 0:
            has_words = True
            # Adjust word timestamps based on chunk start time
            chunk_words = data['words']
            for word in chunk_words:
                # Convert chunk_start_ms from milliseconds to seconds for word timestamp adjustment
                word['start'] = word['start'] + (chunk_start_ms / 1000)
                word['end'] = word['end'] + (chunk_start_ms / 1000)
            words.extend(chunk_words)
    
    # If no words, handle other response formats
    if not has_words:
        logger.warning("No word-level timestamps found in transcription results")
        
        texts = []
        
        for chunk, _ in results:
            # Convert Pydantic model to dict
            data = chunk.model_dump() if hasattr(chunk, 'model_dump') else chunk
            
            # Get text
            if isinstance(data, dict):
                text = data.get('text', '')
            else:
                text = getattr(chunk, 'text', '')
            
            texts.append(text)
        
        merged_text = " ".join(texts)
        return {"text": merged_text}
    
    # Create final transcription from word-level data
    # Sort words by start time to ensure proper ordering
    words.sort(key=lambda x: x['start'])
    
    # Check and add word property if missing
    for i, word_dict in enumerate(words):
        if 'word' not in word_dict and 'text' in word_dict:
            words[i]['word'] = word_dict['text']
    
    # Build the final text
    final_text = ' '.join(word.get('word', '') for word in words)
    
    return {
        "text": final_text,
        "words": words
    }

def transcribe_single_chunk(client: Groq, chunk: AudioSegment, chunk_num: int, total_chunks: int) -> tuple[dict, float]:
    """
    Transcribe a single audio chunk with Groq API.
    
    Args:
        client: Groq client instance
        chunk: Audio segment to transcribe
        chunk_num: Current chunk number
        total_chunks: Total number of chunks
        
    Returns:
        Tuple of (transcription result, processing time)
    """
    total_api_time = 0
    temp_file_path = None
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.flac', delete=False) as temp_file:
            temp_file_path = temp_file.name
            chunk.export(temp_file_path, format='flac')
        
        while True:
            start_time = time.time()
            try:
                with open(temp_file_path, "rb") as audio_file:
                    result = client.audio.transcriptions.create(
                        file=(os.path.basename(temp_file_path), audio_file, "audio/flac"),
                        model=args.model,
                        language=args.language,
                        response_format="verbose_json",
                        temperature=0,
                        timestamp_granularities=["word"]  # Get word-level timestamps
                    )
                api_time = time.time() - start_time
                total_api_time += api_time
                
                logger.info(f"Chunk {chunk_num}/{total_chunks} processed in {api_time:.2f}s")
                return result, total_api_time
                
            except RateLimitError as e:
                logger.warning(f"Rate limit hit for chunk {chunk_num} - retrying in 60 seconds...")
                time.sleep(60)  # default wait time
                continue
                
            except Exception as e:
                logger.error(f"Error transcribing chunk {chunk_num}: {str(e)}")
                raise
    
    finally:
        # Clean up temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_file_path}: {e}")

def transcribe_audio_in_chunks(file_path: str) -> dict:
    """
    Transcribe audio in chunks with overlap with Whisper via Groq API.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        dict: Containing transcription results
    """
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY environment variable not set")
        raise ValueError("GROQ_API_KEY environment variable not set")
    
    file_path = Path(file_path)
    logger.info(f"Starting transcription of: {file_path}")
    client = Groq(api_key=api_key, max_retries=0)
    
    converted_file = None
    try:
        # Convert to appropriate format if needed
        if not args.no_convert:
            file_extension = file_path.suffix.lower()
            if file_extension == '.wav':
                # Check if WAV needs re-encoding
                audio = AudioSegment.from_file(file_path)
                if not check_audio_format(audio):
                    logger.info("WAV file needs re-encoding to meet requirements")
                    converted_file = convert_to_pcm(file_path) if args.use_pcm else convert_to_flac(file_path)
                    file_path = converted_file
            elif file_extension == '.flac' and not args.use_pcm:
                # Check if FLAC needs re-encoding
                audio = AudioSegment.from_file(file_path)
                if not check_audio_format(audio):
                    logger.info("FLAC file needs re-encoding to meet requirements")
                    converted_file = convert_to_flac(file_path)
                    file_path = converted_file
            else:
                # Convert other formats
                converted_file = convert_to_pcm(file_path) if args.use_pcm else convert_to_flac(file_path)
                file_path = converted_file
        
        # Load the audio and get info
        try:
            audio = AudioSegment.from_file(file_path)
        except Exception as e:
            logger.error(f"Failed to load audio: {str(e)}")
            raise RuntimeError(f"Failed to load audio: {str(e)}")
        
        duration = len(audio)
        logger.info(f"Audio duration: {duration/1000:.2f}s")
        
        # Check if we need to chunk based on duration (10 minutes = 600 seconds)
        if duration > 600 * 1000:  # Convert seconds to milliseconds
            logger.info("Audio duration exceeds 10 minutes, using chunking...")
            # Calculate # of chunks
            chunk_ms = args.chunk_length * 1000
            overlap_ms = args.overlap * 1000
            total_chunks = ((duration - overlap_ms) // (chunk_ms - overlap_ms)) + 1
            logger.info(f"Processing {total_chunks} chunks...")
            
            results = []
            total_transcription_time = 0

            # Loop through each chunk, extract current chunk from audio, transcribe    
            for i in range(total_chunks):
                start = i * (chunk_ms - overlap_ms)
                end = min(start + chunk_ms, duration)
                    
                logger.info(f"Processing chunk {i+1}/{total_chunks}")
                logger.info(f"Time range: {start/1000:.1f}s - {end/1000:.1f}s")
                    
                chunk = audio[start:end]
                result, chunk_time = transcribe_single_chunk(client, chunk, i+1, total_chunks)
                total_transcription_time += chunk_time
                results.append((result, start))
                
            # Merge the results
            final_result = merge_transcripts(results, args.overlap)
            logger.info(f"Total Groq API transcription time: {total_transcription_time:.2f}s")
            
            return final_result
        else:
            # For short audio, transcribe directly
            logger.info("Audio duration under 10 minutes, processing in one chunk...")
            result, total_time = transcribe_single_chunk(client, audio, 1, 1)
            logger.info(f"Total Groq API transcription time: {total_time:.2f}s")
            return result
    
    finally:
        # Delete temporary file unless --keep-flac is specified
        if converted_file and not args.keep_flac:
            try:
                # Add a small delay to ensure file is released by OS
                time.sleep(0.5)
                if isinstance(converted_file, Path):
                    converted_file.unlink(missing_ok=True)
                else:
                    os.unlink(converted_file)
                logger.info(f"Deleted temporary file: {converted_file}")
            except Exception as e:
                logger.error(f"Error deleting temporary file: {e}")

def main():
    global args
    args = get_args()

    # Setup logging
    setup_logger(args.debug, args.verbose)
    if args.debug:
        logger.info("Debug mode enabled")

    # Initialize an empty dictionary to store the files
    files_dict = {}

    # Check if args.audio_path is a directory
    if os.path.isdir(args.audio_path):
        logger.info("Directory found.")
        for root, dirs, files in os.walk(args.audio_path):
            for file in files:
                if file.endswith(('.mp3', '.wav', '.ogg', '.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4a', '.flac', '.aac', '.wma', '.aiff')):
                    files_dict[file] = os.path.join(root, file)
    # Check if args.audio_path is a file
    elif os.path.isfile(args.audio_path):
        normalized_file = os.path.normpath(args.audio_path)
        files_dict[normalized_file] = normalized_file
        logger.info("File found.")
    # Check if args.audio_path is a wildcard pattern
    elif '*' in args.audio_path or '?' in args.audio_path:
        logger.info("Wildcard pattern found.")
        for file in glob.glob(args.audio_path):
            files_dict[file] = file
    else:
        logger.error("Invalid input. Please provide a valid file, directory, or wildcard pattern.")
        return

    for file_name, file_path in files_dict.items():
        logger.info(f"Processing file: {file_name}")

        if not os.path.exists(file_path):
            logger.error(f"Audio File {file_name} does not exist!")
            continue

        full_file_name = os.path.basename(file_path)
        file_name_without_ext, file_extension = os.path.splitext(full_file_name)
        file_dir = os.path.dirname(file_path)

        # Check if transcript exists
        if check_transcript_exists(file_dir, file_name_without_ext) and not args.force:
            logger.info(f"Transcript for {file_name_without_ext} exists! Using existing JSON to generate SRT and text files.")
            json_file = os.path.join(file_dir, f"{file_name_without_ext}.json")
            with open(json_file, 'r', encoding='utf-8') as f:
                response_data = json.load(f)
            
            # Create text file
            text_file = os.path.join(file_dir, f"{file_name_without_ext}.txt")
            create_text_file(response_data['words'], text_file)

            # Create SRT file
            srt_file = os.path.join(file_dir, f"{file_name_without_ext}.srt")
            if args.davinci_srt:
                create_davinci_srt(response_data['words'], srt_file, args.silentportions, args.padding,
                                   args.fps, args.fps_offset_start, args.fps_offset_end)
            elif args.word_srt:
                create_word_level_srt(response_data['words'], srt_file, remove_fillers=args.remove_fillers, 
                                     filler_words=FILLER_WORDS, fps=args.fps, 
                                     offset_frame_start=args.fps_offset_start, 
                                     offset_frame_end=args.fps_offset_end)
            else:
                create_srt(response_data['words'], srt_file, args.chars_per_line, args.silentportions,
                          args.fps, args.fps_offset_start, args.fps_offset_end)
            continue

        # Change to file directory
        original_dir = os.getcwd()
        if file_dir:
            os.chdir(file_dir)
            logger.info(f"Working directory changed to: {file_dir}")

        # Transcribe using Groq API
        try:
            response_data = transcribe_audio_in_chunks(file_path)
            if not response_data:
                logger.error(f"Failed to transcribe {file_name}")
                continue
                
            # Save JSON response
            json_file = os.path.join(file_dir, f"{file_name_without_ext}.json")
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, ensure_ascii=False, indent=2)
            logger.info(f"JSON response saved to {json_file}")

            # Create text file
            text_file = os.path.join(file_dir, f"{file_name_without_ext}.txt")
            create_text_file(response_data['words'], text_file)

            # Create SRT file
            srt_file = os.path.join(file_dir, f"{file_name_without_ext}.srt")
            if args.davinci_srt:
                create_davinci_srt(response_data['words'], srt_file, args.silentportions, args.padding,
                                   args.fps, args.fps_offset_start, args.fps_offset_end)
            elif args.word_srt:
                create_word_level_srt(response_data['words'], srt_file, remove_fillers=args.remove_fillers, 
                                     filler_words=FILLER_WORDS, fps=args.fps, 
                                     offset_frame_start=args.fps_offset_start, 
                                     offset_frame_end=args.fps_offset_end)
            else:
                create_srt(response_data['words'], srt_file, args.chars_per_line, args.silentportions,
                          args.fps, args.fps_offset_start, args.fps_offset_end)

            logger.info(f"Transcription completed for {file_name}")
            
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            continue
        
        finally:
            # Change back to original directory
            if file_dir:
                os.chdir(original_dir)

if __name__ == '__main__':
    main() 