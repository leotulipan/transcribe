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
# ]
# ///

# based on https://github.com/groq/groq-api-cookbook/blob/main/tutorials/audio-chunking/audio_chunking_code.py
# example call
# set GROQ_API_KEY in .env
#
#  uv run --link-mode=copy .\groq_audio_chunking_adapted.py -l de .\audio-test.mkv
#

from groq import Groq, RateLimitError
from pydub import AudioSegment
import json
from pathlib import Path
from datetime import datetime
import time
import subprocess
import os
import tempfile
import re
import argparse
from dotenv import load_dotenv
from pydub import AudioSegment
import librosa
import textwrap
from datetime import timedelta
import sys

def get_args():
    parser = argparse.ArgumentParser(description="Audio chunking and transcription using Groq API.")
    parser.add_argument("audio_path", type=str, help="Path to the input audio file or segments JSON file.")
    parser.add_argument("--language", "-l", type=str, default="de", help="Language of the audio (default: en).")
    parser.add_argument("--model", "-m",type=str, default="whisper-large-v3", help="Model to use for transcription (default: whisper-large-v3).")
    parser.add_argument("--chunk_length", "-c", type=int, default=600, help="Length of each chunk in seconds (default: 600).")
    parser.add_argument("--overlap", "-o", type=int, default=10, help="Overlap between chunks in seconds (default: 10).")
    parser.add_argument("--srt-only", "-s", action="store_true", help="Convert existing segments JSON to SRT format only.")
    return parser.parse_args()

args = get_args()


def preprocess_audio_with_ffmpeg(input_path: Path) -> Path:
    """
    Preprocess audio file to 16kHz mono FLAC using ffmpeg.
    FLAC provides lossless compression for faster upload times.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    with tempfile.NamedTemporaryFile(suffix='.flac', delete=False) as temp_file:
        output_path = Path(temp_file.name)
        
    print("Converting audio to 16kHz mono FLAC...")
    try:
        subprocess.run([
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'error',
            '-i', input_path,
            '-ar', '16000',
            '-ac', '1',
            '-c:a', 'flac',
            '-y',
            output_path
        ], check=True) 
        return output_path
    # We'll raise an error if our FFmpeg conversion fails
    except subprocess.CalledProcessError as e:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"FFmpeg conversion failed: {e.stderr}")

def preprocess_audio(input_path: Path) -> Path:
    """
    Preprocess audio file to 16kHz mono FLAC using pydub.
    FLAC provides lossless compression for faster upload times.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    with tempfile.NamedTemporaryFile(suffix='.flac', delete=False) as temp_file:
        output_path = Path(temp_file.name)
        
    print("Converting audio to 16kHz mono FLAC...")
    try:
        # Load audio file
        audio = AudioSegment.from_file(input_path)
        # Set frame rate to 16kHz and channels to mono
        audio = audio.set_frame_rate(16000).set_channels(1)
        # Export as FLAC
        audio.export(output_path, format="flac")
        return output_path
    except Exception as e:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"Audio conversion failed: {e}")        
    
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

    Raises:
        Exception: If chunk transcription fails after retries
    """
    total_api_time = 0
    
    while True:
        with tempfile.NamedTemporaryFile(suffix='.flac', delete=False) as temp_file:
            chunk.export(temp_file.name, format='flac')
            
            start_time = time.time()
            try:
                result = client.audio.transcriptions.create(
                    file=("chunk.flac", temp_file, "audio/flac"),
                    model=args.model,
                    language=args.language, # We highly recommend specifying the language of your audio if you know it
                    response_format="verbose_json",
                    temperature=0,  # For best transcription quality
                    timestamp_granularities=["segment"]  # Get segment-level timestamps
                )
                api_time = time.time() - start_time
                total_api_time += api_time
                
                print(f"Chunk {chunk_num}/{total_chunks} processed in {api_time:.2f}s")
                return result, total_api_time
                os.remove(temp_file.name)
                
            except RateLimitError as e:
                print(f"\nRate limit hit for chunk {chunk_num} - retrying in 60 seconds...")
                time.sleep(60)  # default wait time
                continue
                
            except Exception as e:
                print(f"Error transcribing chunk {chunk_num}: {str(e)}")
                raise

def find_longest_common_sequence(sequences: list[str], match_by_words: bool = True) -> str:
    """
    Find the optimal alignment between sequences with longest common sequence and sliding window matching.
    
    Args:
        sequences: List of text sequences to align and merge
        match_by_words: Whether to match by words (True) or characters (False)
        
    Returns:
        str: Merged sequence with optimal alignment
        
    Raises:
        RuntimeError: If there's a mismatch in sequence lengths during comparison
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
                raise RuntimeError(
                    "Mismatched subsequences detected during transcript merging."
                )

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
    Merge transcription chunks and handle overlaps by:
    1. Adjust all segment timestamps based on the chunk's position in the audio
    2. Merge all segments within each chunk's overlap/stride
    3. Merge chunk boundaries using find_longest_common_sequence
    
    Args:
        results: List of (result, start_time) tuples
        overlap: Overlap between chunks in seconds
        
    Returns:
        dict: Merged transcription
    """
    print("\nMerging results...")
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

def convert_to_srt(result: dict, output_path: Path) -> None:
    """
    Convert Groq's verbose JSON output to SRT format with metadata-based filtering.
    
    Args:
        result: Transcription result dictionary from Groq API
        output_path: Path to save the SRT file
    """
    def format_time(seconds: float) -> str:
        """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
        td = timedelta(seconds=seconds)
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int(td.microseconds / 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def split_text_into_chunks(text: str, max_chars: int = 80) -> list[str]:
        """Split text into chunks of maximum length while respecting word boundaries"""
        return textwrap.wrap(text, width=max_chars, break_long_words=False)

    # Filter segments based on metadata quality indicators
    filtered_segments = []
    for segment in result['segments']:
        # Skip segments with:
        # 1. High no_speech_prob (likely non-speech)
        # 2. Very negative avg_logprob (low confidence)
        # 3. Unusual compression_ratio (potential issues)
        # 4. Zero start time (unless it's the only segment)
        if (segment['no_speech_prob'] < 0.5 and  # Less than 50% chance of being non-speech
            segment['avg_logprob'] > -0.5 and    # Better than -0.5 log probability
            0.8 < segment['compression_ratio'] < 2.0 and  # Normal speech patterns
            (segment['start'] != 0 or all(s['start'] == 0 for s in result['segments']))):
            filtered_segments.append(segment)

    # Sort segments by start time to ensure proper ordering
    filtered_segments.sort(key=lambda x: x['start'])

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
    srt_path = output_path.with_suffix('.srt')
    with open(srt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(srt_lines))
    
    print(f"SRT file saved to: {srt_path}")

def save_results(result: dict, audio_path: Path) -> Path:
    """
    Save transcription results to files.
    
    Args:
        result: Transcription result dictionary
        audio_path: Original audio file path
        
    Returns:
        base_path: Base path where files were saved

    Raises:
        IOError: If saving results fails
    """
    try:
        output_dir = Path("transcriptions")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = output_dir / f"{Path(audio_path).stem}_{timestamp}"
        
        # Save results in different formats
        with open(f"{base_path}.{args.language}.txt", 'w', encoding='utf-8') as f:
            f.write(result["text"])
        
        with open(f"{base_path}_full.{args.language}.json", 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        with open(f"{base_path}_segments.{args.language}.json", 'w', encoding='utf-8') as f:
            json.dump(result["segments"], f, indent=2, ensure_ascii=False)
        
        # Convert to SRT format
        convert_to_srt(result, base_path)
        
        print(f"\nResults saved to transcriptions folder:")
        print(f"- {base_path}.{args.language}.txt")
        print(f"- {base_path}_full.{args.language}.json")
        print(f"- {base_path}_segments.{args.language}.json")
        print(f"- {base_path}.{args.language}.srt")
        
        return base_path
    
    except IOError as e:
        print(f"Error saving results: {str(e)}")
        raise

def transcribe_audio_in_chunks(audio_path: Path, chunk_length: int = 600, overlap: int = 10) -> dict:
    """
    Transcribe audio in chunks with overlap with Whisper via Groq API.
    
    Args:
        audio_path: Path to audio file
        chunk_length: Length of each chunk in seconds
        overlap: Overlap between chunks in seconds
    
    Returns:
        dict: Containing transcription results
    
    Raises:
        ValueError: If Groq API key is not set
        RuntimeError: If audio file fails to load
    """
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    
    print(f"\nStarting transcription of: {audio_path}")
    # Make sure your Groq API key is configured. If you don't have one, you can get one at https://console.groq.com/keys!
    client = Groq(api_key=api_key, max_retries=0)
    
    processed_path = None
    try:
        # Preprocess audio and get basic info
        processed_path = preprocess_audio(audio_path)
        try:
            audio = AudioSegment.from_file(processed_path, format="flac")
        except Exception as e:
            raise RuntimeError(f"Failed to load audio: {str(e)}")
        
        duration = len(audio)
        print(f"Audio duration: {duration/1000:.2f}s")
        
        # Calculate # of chunks
        chunk_ms = chunk_length * 1000
        overlap_ms = overlap * 1000
        total_chunks = (duration // (chunk_ms - overlap_ms)) + 1
        print(f"Processing {total_chunks} chunks...")
        
        results = []
        total_transcription_time = 0

        # Loop through each chunk, extract current chunk from audio, transcribe    
        for i in range(total_chunks):
            start = i * (chunk_ms - overlap_ms)
            end = min(start + chunk_ms, duration)
                
            print(f"\nProcessing chunk {i+1}/{total_chunks}")
            print(f"Time range: {start/1000:.1f}s - {end/1000:.1f}s")
                
            chunk = audio[start:end]
            result, chunk_time = transcribe_single_chunk(client, chunk, i+1, total_chunks)
            total_transcription_time += chunk_time
            results.append((result, start))
            
        final_result = merge_transcripts(results, overlap)
        save_results(final_result, audio_path)
            
        print(f"\nTotal Groq API transcription time: {total_transcription_time:.2f}s")
        
        return final_result
    
    # Clean up temp files regardless of successful creation    
    finally:
        if processed_path:
            Path(processed_path).unlink(missing_ok=True)

def convert_segments_to_srt(segments_path: Path, language: str = "en") -> None:
    """
    Convert an existing segments JSON file to SRT format.
    
    Args:
        segments_path: Path to the segments JSON file or original media file
        language: Language code for the output filename
    """
    try:
        # If the input path is not a segments file, try to find it
        if not str(segments_path).endswith(f"_segments.{language}.json"):
            # Try to find the segments file in the transcriptions directory
            possible_paths = [
                # Try with _segments suffix in same directory
                segments_path.parent / f"{segments_path.stem}_segments.{language}.json",
                # Try in transcriptions directory
                Path("transcriptions") / f"{segments_path.stem}_segments.{language}.json",
                # Try with timestamp pattern in transcriptions directory
                Path("transcriptions") / f"{segments_path.stem}_*_segments.{language}.json"
            ]
            
            segments_path = None
            for path in possible_paths:
                if path.is_file():
                    segments_path = path
                    break
                elif "*" in str(path):
                    # Handle glob pattern
                    matches = list(Path(path.parent).glob(path.name))
                    if matches:
                        segments_path = matches[0]  # Use the first match
                        break
            
            if not segments_path:
                print("Error: Could not find segments file. Tried:")
                for path in possible_paths:
                    print(f"- {path}")
                sys.exit(1)
        
        print(f"Reading segments file: {segments_path}")
        with open(segments_path, 'r', encoding='utf-8') as f:
            segments_data = json.load(f)
        
        # Create a result dictionary in the format expected by convert_to_srt
        result = {
            "text": " ".join(segment["text"] for segment in segments_data),
            "segments": segments_data
        }
        
        # Use the same directory as the segments file
        output_path = segments_path.parent / segments_path.stem.replace("_segments", "")
        convert_to_srt(result, output_path)
        
    except Exception as e:
        print(f"Error converting segments to SRT: {str(e)}")
        raise

if __name__ == "__main__":
    args = get_args()
    
    if args.srt_only:
        # Handle SRT-only conversion
        input_path = Path(args.audio_path)
        convert_segments_to_srt(input_path, args.language)
    else:
        # Handle full transcription process
        transcribe_audio_in_chunks(Path(args.audio_path), args.chunk_length, args.overlap)
