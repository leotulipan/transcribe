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
import base64
from typing import Union

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

def transcribe_single_chunk(client: Groq, chunk: Union[AudioSegment, str, Path], chunk_num: int, total_chunks: int) -> tuple[dict, float]:
    """
    Transcribe a single audio chunk using Groq API.
    
    Args:
        client: Groq client
        chunk: AudioSegment chunk or path to audio file
        chunk_num: Current chunk number
        total_chunks: Total number of chunks
        
    Returns:
        tuple: (Result dict, time taken)
    """
    global args
    model = args.model
    logger.info(f"Using Groq model: {model}")
    
    temp_file = None
    if isinstance(chunk, (str, Path)):
        # This is already a file path, use it directly
        chunk_path = chunk
    else:
        # This is an AudioSegment, export it to temp file
        temp_file = tempfile.NamedTemporaryFile(suffix=".flac", delete=False)
        chunk_path = temp_file.name
        temp_file.close()
        
        try:
            chunk.export(chunk_path, format="flac")
        except Exception as e:
            logger.error(f"Failed to export audio chunk: {e}")
            if os.path.exists(chunk_path):
                os.unlink(chunk_path)
            raise
    
    try:
        # Encode audio file to base64
        with open(chunk_path, "rb") as audio_file:
            audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")
            
        # Prepare system prompt
        system_prompt = f"""
You are a professional audio transcription service. Your task is to convert the given audio to text with high accuracy.
Include proper spacing, punctuation, and line breaks. The transcription should respect sentence boundaries.
Only transcribe the actual speech content, without adding any additional commentary.
Output must be a valid JSON object with the following structure:
{{
    "text": "The full transcript text goes here.",
    "words": [
        {{
            "text": "word",
            "start": start time in seconds,
            "end": end time in seconds
        }},
    ]
}}
"""
        
        user_prompt = f"""
Please transcribe this audio file to text.
Audio timestamp format: seconds
Output format: JSON as specified
Language: {args.language}
"""
        
        # Prepare the transcription request
        transcription_start_time = time.time()
        
        # Number of retries
        max_retries = 3
        retry_count = 0
        retry_delay = 5  # seconds
        
        while retry_count < max_retries:
            try:
                chat_completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "audio", "audio_data": audio_base64}
                        ]}
                    ],
                    max_tokens=4000
                )
                
                transcription_time = time.time() - transcription_start_time
                logger.info(f"Groq API call completed in {transcription_time:.2f}s (chunk {chunk_num}/{total_chunks})")
                
                # Extract and parse response content
                response_content = chat_completion.choices[0].message.content
                
                # Parse the JSON response
                try:
                    import re
                    # Extract JSON using regex in case of extra text
                    json_match = re.search(r'```json\s*(.*?)\s*```', response_content, re.DOTALL)
                    if json_match:
                        result = json.loads(json_match.group(1))
                    else:
                        # Try to extract JSON using normal pattern
                        json_start = response_content.find('{')
                        json_end = response_content.rfind('}') + 1
                        if json_start >= 0 and json_end > json_start:
                            json_str = response_content[json_start:json_end]
                            result = json.loads(json_str)
                        else:
                            logger.warning("Could not find JSON structure in response, using whole response")
                            result = {"text": response_content, "words": []}
                except Exception as e:
                    logger.error(f"Failed to parse JSON response: {e}")
                    logger.debug(f"Response content: {response_content}")
                    # Return a basic dictionary with the raw text
                    result = {"text": response_content, "words": []}
                
                return result, transcription_time
                
            except RateLimitError as e:
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"Rate limit error after {max_retries} retries. Giving up.")
                    raise
                else:
                    logger.warning(f"Rate limit error. Retrying in {retry_delay} seconds... ({retry_count}/{max_retries})")
                    time.sleep(retry_delay)
                    # Increase delay for next retry
                    retry_delay *= 2
            except Exception as e:
                logger.error(f"Transcription error: {str(e)}")
                raise
                
    finally:
        # Clean up temporary file if we created one
        if temp_file and os.path.exists(chunk_path):
            try:
                os.unlink(chunk_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file: {e}")

def transcribe_audio_in_chunks(file_path: str) -> dict:
    """
    Transcribe audio file in chunks using Groq API.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        dict: Transcription result with text and words
    """
    global args
    
    if args.debug:
        logger.info(f"Current working directory: {os.getcwd()}")

    # Initialize Groq client
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    # Check if audio file exists
    if not os.path.exists(file_path):
        logger.error(f"Audio file not found: {file_path}")
        sys.exit(1)

    # Get audio file information
    file_length_secs = check_audio_length(file_path)
    logger.info(f"Audio length: {file_length_secs} seconds")

    # Check if audio file is not too long
    if file_length_secs > MAX_AUDIO_LENGTH:
        logger.error(f"Audio file too long: {file_length_secs} seconds (max: {MAX_AUDIO_LENGTH} seconds)")
        sys.exit(1)

    # Check file format and convert if necessary
    original_file = file_path
    is_converted = False

    if not args.no_convert:
        logger.info("Checking audio format...")
        format_ok = check_audio_format(file_path, target_format="flac")

        if not format_ok:
            logger.info("Converting audio to FLAC format...")
            if args.use_pcm:
                logger.info("Using PCM format as requested")
                new_file_path = convert_to_pcm(file_path)
            else:
                logger.info("Using FLAC format")
                new_file_path = convert_to_flac(file_path)

            if new_file_path:
                file_path = new_file_path
                is_converted = True
                logger.info(f"Converted audio file: {file_path}")
            else:
                logger.error("Failed to convert audio file")
                sys.exit(1)
    else:
        logger.info("Using audio file as-is (no conversion)")

    # Check file size and use chunking if necessary
    file_size_mb = check_file_size(file_path)
    logger.info(f"File size: {file_size_mb:.2f} MB")
    
    # Groq API limit is 25MB
    if file_size_mb > 25:
        logger.info("File size exceeds 25MB limit for Groq API. Using chunking...")
        from pydub import AudioSegment

        # Load audio file
        audio = AudioSegment.from_file(file_path)
        
        # Split audio into chunks
        chunk_length_ms = args.chunk_length * 1000  # Convert seconds to milliseconds
        overlap_ms = args.overlap * 1000  # Convert seconds to milliseconds
        
        # Calculate number of chunks
        duration_ms = len(audio)
        chunks_count = max(1, int((duration_ms - overlap_ms) / (chunk_length_ms - overlap_ms)) + 1)
        logger.info(f"Splitting audio into {chunks_count} chunks (chunk length: {args.chunk_length}s, overlap: {args.overlap}s)")
        
        # Process each chunk
        results = []
        temp_files = []

        for i in range(chunks_count):
            start_ms = i * (chunk_length_ms - overlap_ms)
            end_ms = min(start_ms + chunk_length_ms, duration_ms)
            
            logger.info(f"Processing chunk {i+1}/{chunks_count} ({start_ms/1000:.1f}s to {end_ms/1000:.1f}s)")
            
            # Extract chunk
            chunk = audio[start_ms:end_ms]
            
            # Create temporary file for chunk
            with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as tmp:
                chunk_path = tmp.name
                temp_files.append(chunk_path)
                
            # Export chunk to temporary file
            chunk.export(chunk_path, format="flac")
            
            # Transcribe chunk
            try:
                result, _ = transcribe_single_chunk(client, chunk_path, i+1, chunks_count)
                results.append((result, start_ms))
            except Exception as e:
                logger.error(f"Error transcribing chunk {i+1}: {e}")
                # Continue with next chunk instead of failing completely
                continue
                
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass
                
        # Merge results
        if not results:
            logger.error("No chunks were successfully transcribed")
            sys.exit(1)
            
        merged_result = merge_transcripts(results, args.overlap)
        
        # Clean up converted file if we created it
        if is_converted and not args.keep_flac and os.path.exists(file_path) and file_path != original_file:
            try:
                os.remove(file_path)
                logger.info(f"Removed temporary FLAC file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary FLAC file: {e}")
                
        return merged_result
    else:
        # For smaller files, use single transcription
        logger.info("Using single transcription (file size is under 25MB)")
        result, _ = transcribe_single_chunk(client, file_path, 1, 1)
        
        # Clean up converted file if we created it
        if is_converted and not args.keep_flac and os.path.exists(file_path) and file_path != original_file:
            try:
                os.remove(file_path)
                logger.info(f"Removed temporary FLAC file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary FLAC file: {e}")
                
        return result

def check_json_exists(file_dir, file_name):
    """
    Check if a JSON transcript file already exists for the given file name.
    
    Args:
        file_dir: Directory containing the file
        file_name: Base name of the file without extension
        
    Returns:
        (bool, str): (True, json_path) if JSON exists, (False, "") otherwise
    """
    # Check for Groq-specific JSON
    json_path = os.path.join(file_dir, f"{file_name}_groq.json")
    if os.path.exists(json_path):
        logger.info(f"Found Groq JSON file: {json_path}")
        return True, json_path
    
    # Check for generic JSON as fallback
    json_path = os.path.join(file_dir, f"{file_name}.json")
    if os.path.exists(json_path):
        logger.info(f"Found generic JSON file: {json_path}")
        return True, json_path
        
    return False, ""

def save_results(result, file_path):
    """
    Save transcription results to files.
    
    Args:
        result: Transcription result dict
        file_path: Original audio file path
    """
    file_path = Path(file_path)
    file_dir = file_path.parent
    file_name = file_path.stem
    
    # Save JSON (with API name in filename)
    json_file = file_dir / f"{file_name}_groq.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON saved to: {json_file}")
    
    # Create text file
    text_file = file_dir / f"{file_name}.txt"
    if 'text' in result:
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(result['text'])
        logger.info(f"Text saved to: {text_file}")
    
    # Create SRT file using our helper functions
    srt_file = file_dir / f"{file_name}.srt"
    if 'words' in result and result['words']:
        if args.davinci_srt:
            create_davinci_srt(
                result['words'],
                srt_file,
                silentportions=args.silentportions,
                padding_start=0,
                padding_end=args.padding,
                fps=args.fps,
                fps_offset_start=args.fps_offset_start,
                fps_offset_end=args.fps_offset_end,
                remove_fillers=args.remove_fillers
            )
        elif args.word_srt:
            create_word_level_srt(
                result['words'],
                srt_file,
                remove_fillers=args.remove_fillers,
                fps=args.fps,
                fps_offset_start=args.fps_offset_start,
                fps_offset_end=args.fps_offset_end
            )
        else:
            create_srt(
                result['words'],
                srt_file,
                chars_per_line=args.chars_per_line,
                silentportions=args.silentportions,
                fps=args.fps,
                fps_offset_start=args.fps_offset_start,
                fps_offset_end=args.fps_offset_end,
                padding_end=args.padding,
                remove_fillers=args.remove_fillers
            )
        logger.info(f"SRT saved to: {srt_file}")

def main():
    global args
    args = get_args()
    
    # Setup logging using common helper
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
        file_name, file_extension = os.path.splitext(full_file_name)
        file_dir = os.path.dirname(file_path)
        
        # Check if transcript exists
        if check_transcript_exists(file_dir, file_name) and not args.force:
            logger.info(f"Transcript for {file_name} exists! Skipping.")
            continue
            
        # Check if JSON exists and reuse it if available
        json_exists, json_path = check_json_exists(file_dir, file_name)
        if json_exists and not args.force:
            logger.info(f"JSON file for {file_name} exists! Skipping audio processing and API call.")
            
            # Load the existing JSON
            with open(json_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
                
            # Regenerate output files (text, SRT)
            save_results(result, file_path)
            logger.info(f"Regenerated output files from existing JSON")
            continue
            
        # Process the file
        try:
            result = transcribe_audio_in_chunks(file_path)
            save_results(result, file_path)
            
        except Exception as e:
            logger.error(f"Error processing {file_name}: {e}")
            continue

if __name__ == '__main__':
    main() 