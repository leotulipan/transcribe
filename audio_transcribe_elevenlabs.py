# /// script
# dependencies = [
#   "load_dotenv",
#   "argparse",
#   "requests",
#   "datetime",
#   "pydub",
#   "loguru",
# ]
# ///

import glob
from dotenv import load_dotenv
import os
import pprint
import json
import requests
from datetime import date, timedelta, datetime
import argparse
from pydub import AudioSegment
from loguru import logger

# global variables
args = ""
headers = None
MAX_AUDIO_LENGTH = 7200  # seconds

def setup_logger():
    """Configure loguru logger"""
    logger.remove()  # Remove default handler
    
    # Always log errors to console
    logger.add(lambda msg: print(msg), level="ERROR")
    
    # Add file logging
    logger.add(
        "transcribe_{time}.log",
        rotation="1 day",
        retention="7 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    
    # Add console logging for all levels if verbose
    if args.verbose:
        logger.add(lambda msg: print(msg), level="INFO")

def in_debug_mode():
    if args.debug:
        return True
    return False

def clear_screen():
    return

def check_transcript_exists(file_path, file_name):
    transcript_path = os.path.join(file_path, f"{file_name}.txt")
    srt_path = os.path.join(file_path, f"{file_name}.srt")
    json_path = os.path.join(file_path, f"{file_name}.json")
    return all(os.path.exists(p) for p in [transcript_path, srt_path, json_path])

def check_audio_length(file_path):
    """Check if audio file is shorter than MAX_AUDIO_LENGTH"""
    audio = AudioSegment.from_file(file_path)
    duration_seconds = len(audio) / 1000.0  # Convert milliseconds to seconds
    if duration_seconds > MAX_AUDIO_LENGTH:
        raise RuntimeError(f"Audio duration ({duration_seconds:.1f}s) exceeds maximum allowed length ({MAX_AUDIO_LENGTH}s)")
    return True

def convert_to_pcm(input_file):
    """Convert audio/video file to PCM format (mono, 16-bit, 16kHz)"""
    logger.info(f"Converting {input_file} to PCM format...")
    audio = AudioSegment.from_file(input_file)
    # Convert to mono
    audio = audio.set_channels(1)
    # Set sample rate to 16kHz
    audio = audio.set_frame_rate(16000)
    # Set sample width to 2 bytes (16-bit)
    audio = audio.set_sample_width(2)
    
    # Create output filename
    output_file = os.path.splitext(input_file)[0] + ".wav"
    # Export as PCM WAV
    audio.export(output_file, format="wav", parameters=["-f", "s16le"])
    logger.info(f"PCM conversion completed: {output_file}")
    return output_file

def check_file_size(file_path):
    """Check if file size is under 1000MB"""
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if size_mb > 1000:
        raise RuntimeError(f"File size ({size_mb:.2f}MB) exceeds 1000MB limit")
    return True

def create_srt(words, output_file, chars_per_line=80):
    """Create SRT file from words data"""
    logger.info(f"Creating SRT file: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        counter = 1
        current_text = ""
        current_start = None
        current_end = None
        
        # Handle initial silence
        if words and words[0]['type'] == 'word' and words[0]['start'] > 0 and args.silentportions > 0:
            f.write(f"{counter}\n")
            f.write(f"00:00:00,000 --> {format_time(words[0]['start'])}\n")
            f.write("(...)\n\n")
            counter += 1
        
        for word in words:
            if word['type'] == 'spacing' and args.silentportions > 0:
                duration_ms = (word['end'] - word['start']) * 1000
                if duration_ms >= args.silentportions:
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
            
            if word['type'] == 'word':
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
        
        if current_text:
            f.write(f"{counter}\n")
            f.write(f"{format_time(current_start)} --> {format_time(current_end)}\n")
            f.write(f"{current_text.strip()}\n\n")
    logger.info("SRT file created successfully")

def format_time(seconds):
    """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def create_text_file(words, output_file):
    """Create plain text file from words data"""
    logger.info(f"Creating text file: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        current_speaker = None
        for word in words:
            if word['type'] == 'word':
                speaker = word.get('speaker_id', 'Unknown')
                if speaker != current_speaker:
                    if current_speaker is not None:
                        f.write("\n")
                    f.write(f"Speaker {speaker}: ")
                    current_speaker = speaker
                f.write(word['text'] + " ")
            elif word['type'] == 'audio_event':
                f.write(f"({word['text']}) ")
    logger.info("Text file created successfully")

def handle_error_response(response):
    """Pretty print error response from API"""
    try:
        error_data = response.json()
        detail = error_data.get('detail', {})
        status = detail.get('status', 'unknown')
        message = detail.get('message', 'No message provided')
        logger.error(f"API Error: {status} - {message}")
    except:
        logger.error(f"Raw error response: {response.text}")

def main():
    global args, headers
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", help="Debug mode", action="store_true")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-f", "--file", "--folder", help="filename, foldername, or pattern to transcribe")
    parser.add_argument("-c", "--chars_per_line", type=int, default=80, help="Maximum characters per line in SRT file")
    parser.add_argument("-s", "--speaker_labels", help="Use this flag to remove speaker labels", action="store_false", default=True)
    parser.add_argument("--keep-flac", help="Keep the generated FLAC file after processing", action="store_true")
    parser.add_argument("--no-convert", help="Send the audio file as-is without conversion", action="store_true")
    parser.add_argument("-l", "--language", help="Language code (ISO-639-1 or ISO-639-3). Examples: en (English), fr (French), de (German)", default=None)
    parser.add_argument("-v", "--verbose", help="Show all log messages in console", action="store_true")
    parser.add_argument("--force", help="Force re-transcription even if files exist", action="store_true")
    parser.add_argument("-p", "--silentportions", type=int, help="Mark pauses longer than X milliseconds with (...)", default=0)
    
    args = parser.parse_args()
    pp = pprint.PrettyPrinter(indent=4)

    # Setup logging
    setup_logger()
    if in_debug_mode():
        logger.add(lambda msg: print(msg), level="DEBUG")
        logger.info("Debug mode enabled")

    load_dotenv()
    headers = {
        "xi-api-key": os.getenv("ELEVENLABS_API_KEY"),
        "Accept": "application/json"
    }

    # Initialize an empty dictionary to store the files
    files_dict = {}

    # Check if args.file is a directory
    if os.path.isdir(args.file):
        logger.info("Directory found.")
        for root, dirs, files in os.walk(args.file):
            for file in files:
                if file.endswith(('.mp3', '.wav', '.ogg', '.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4a', '.flac', '.aac', '.wma', '.aiff')):
                    files_dict[file] = os.path.join(root, file)
    # Check if args.file is a file
    elif os.path.isfile(args.file):
        normalized_file = os.path.normpath(args.file)
        files_dict[normalized_file] = normalized_file
        logger.info("File found.")
    # Check if args.file is a wildcard pattern
    elif '*' in args.file or '?' in args.file:
        logger.info("Wildcard pattern found.")
        for file in glob.glob(args.file):
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
            logger.info(f"Transcript for {file_name} exists! Using existing JSON to generate SRT and text files.")
            json_file = os.path.join(file_dir, f"{file_name}.json")
            with open(json_file, 'r', encoding='utf-8') as f:
                response_data = json.load(f)
            
            # Create text file
            text_file = os.path.join(file_dir, f"{file_name}.txt")
            create_text_file(response_data['words'], text_file)

            # Create SRT file
            srt_file = os.path.join(file_dir, f"{file_name}.srt")
            create_srt(response_data['words'], srt_file, args.chars_per_line)
            continue

        # Convert to FLAC if not already and conversion is not disabled
        if not args.no_convert and file_extension.lower() != '.wav':
            file_path = convert_to_pcm(file_path)

        # Check file size and duration
        try:
            check_file_size(file_path)
            check_audio_length(file_path)
        except RuntimeError as e:
            logger.error(f"Error: {e}")
            continue

        # Change to file directory
        if file_dir:
            os.chdir(file_dir)
            logger.info(f"Working directory changed to: {file_dir}")

        # Transcribe using ElevenLabs API
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        logger.info(f"Starting ElevenLabs transcription... (File size: {file_size_mb:.2f}MB)")

        try:
            with open(file_path, 'rb') as audio_file:
                files = {
                    'file': ('audio.wav', audio_file, 'audio/wav')
                }
                data = {
                    'model_id': 'scribe_v1',
                    'language_code': args.language,
                    'tag_audio_events': 'true',
                    'num_speakers': '2' if args.speaker_labels else None,
                    'timestamps_granularity': 'word',
                    'diarize': 'true' if args.speaker_labels else 'false',
                    'file_format': 'pcm_s16le_16'
                }
                
                # Remove None values from data
                data = {k: v for k, v in data.items() if v is not None}
                
                response = requests.post(
                    "https://api.elevenlabs.io/v1/speech-to-text",
                    headers=headers,
                    files=files,
                    data=data
                )
                
                response.raise_for_status()
                response_data = response.json()

                # Save JSON response
                json_file = os.path.join(file_dir, f"{file_name}.json")
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(response_data, f, ensure_ascii=False, indent=2)
                logger.info(f"JSON response saved to {json_file}")

                # Create text file
                text_file = os.path.join(file_dir, f"{file_name}.txt")
                create_text_file(response_data['words'], text_file)

                # Create SRT file
                srt_file = os.path.join(file_dir, f"{file_name}.srt")
                create_srt(response_data['words'], srt_file, args.chars_per_line)

                logger.info(f"Transcription completed for {file_name}")
                if file_dir:
                    logger.info(f"Files saved in: {file_dir}")

                # Delete FLAC file by default unless --keep-flac is specified
                if not args.keep_flac and file_path.endswith('.wav'):
                    try:
                        os.remove(file_path)
                        logger.info(f"Deleted temporary WAV file: {file_path}")
                    except Exception as e:
                        logger.error(f"Error deleting WAV file: {e}")

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 400:
                handle_error_response(e.response)
            else:
                logger.error(f"HTTP Error: {e}")
            continue
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            continue

if __name__ == '__main__':
    main() 