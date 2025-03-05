# /// script
# dependencies = [
#   "load_dotenv",
#   "argparse",
#   "requests",
#   "datetime",
#   "elevenlabs",
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
from elevenlabs import ElevenLabs
from loguru import logger

# global variables
args = ""
client = None

def setup_logger():
    """Configure loguru logger"""
    logger.remove()  # Remove default handler
    logger.add(
        "transcribe_{time}.log",
        rotation="1 day",
        retention="7 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    logger.add(lambda msg: print(msg), level="INFO")  # Also print to console

def in_debug_mode():
    if args.debug:
        return True
    return False

def clear_screen():
    return

def check_transcript_exists(file_path, file_name):
    transcript_path = os.path.join(file_path, f"{file_name}.txt")
    srt_path = os.path.join(file_path, f"{file_name}.srt")
    return os.path.exists(transcript_path) or os.path.exists(srt_path)

def convert_to_flac(input_file):
    """Convert audio file to FLAC format (mono, 16-bit)"""
    logger.info(f"Converting {input_file} to FLAC format...")
    audio = AudioSegment.from_file(input_file)
    # Convert to mono
    audio = audio.set_channels(1)
    # Set sample rate to 16kHz (standard for speech)
    audio = audio.set_frame_rate(16000)
    # Set sample width to 2 bytes (16-bit)
    audio = audio.set_sample_width(2)
    
    # Create output filename
    output_file = os.path.splitext(input_file)[0] + ".flac"
    # Export as FLAC
    audio.export(output_file, format="flac")
    logger.info(f"FLAC conversion completed: {output_file}")
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
        
        for word in words:
            if word['type'] == 'word':
                if current_start is None:
                    current_start = word['start']
                current_end = word['end']
                current_text += word['text'] + " "
                
                # Check if we need to split due to character limit
                if len(current_text.strip()) >= chars_per_line:
                    # Write current segment
                    f.write(f"{counter}\n")
                    f.write(f"{format_time(current_start)} --> {format_time(current_end)}\n")
                    f.write(f"{current_text.strip()}\n\n")
                    
                    # Reset for next segment
                    counter += 1
                    current_text = ""
                    current_start = None
                    current_end = None
        
        # Write any remaining text
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
        logger.error("API Error Response:")
        logger.error(json.dumps(error_data, indent=2))
    except:
        logger.error(f"Raw error response: {response.text}")

def main():
    global args, client
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", help="Debug mode", action="store_true")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-f", "--file", "--folder", help="filename, foldername, or pattern to transcribe")
    parser.add_argument("-c", "--chars_per_line", type=int, default=80, help="Maximum characters per line in SRT file")
    parser.add_argument("-s", "--speaker_labels", help="Use this flag to remove speaker labels", action="store_false", default=True)
    parser.add_argument("--delete-flac", help="Delete the generated FLAC file after processing", action="store_true")
    parser.add_argument("--no-convert", help="Send the audio file as-is without conversion", action="store_true")
    parser.add_argument("-l", "--language", help="Language code (ISO-639-1 or ISO-639-3). Examples: en (English), fr (French), de (German)", default=None)
    
    args = parser.parse_args()
    pp = pprint.PrettyPrinter(indent=4)

    # Setup logging
    setup_logger()
    if in_debug_mode():
        logger.add(lambda msg: print(msg), level="DEBUG")
        logger.info("Debug mode enabled")

    load_dotenv()
    client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

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
        if check_transcript_exists(file_dir, file_name):
            logger.info(f"Transcript for {file_name} exists!")
            continue

        # Convert to FLAC if not already and conversion is not disabled
        if not args.no_convert and file_extension.lower() != '.flac':
            file_path = convert_to_flac(file_path)

        # Check file size
        try:
            check_file_size(file_path)
        except RuntimeError as e:
            logger.error(f"Error: {e}")
            continue

        # Change to file directory
        if file_dir:
            os.chdir(file_dir)

        # Transcribe using ElevenLabs
        logger.info("Starting ElevenLabs transcription...")

        try:
            response = client.speech_to_text.convert(
                model_id="scribe_v1",
                file=file_path,
                language_code=args.language,
                tag_audio_events=True,
                num_speakers=2 if args.speaker_labels else None,
                timestamps_granularity="word",
                diarize=args.speaker_labels
            )

            # Save JSON response
            json_file = os.path.join(file_dir, f"{file_name}.json")
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(response, f, ensure_ascii=False, indent=2)
            logger.info(f"JSON response saved to {json_file}")

            # Create text file
            text_file = os.path.join(file_dir, f"{file_name}.txt")
            create_text_file(response['words'], text_file)

            # Create SRT file
            srt_file = os.path.join(file_dir, f"{file_name}.srt")
            create_srt(response['words'], srt_file, args.chars_per_line)

            logger.info(f"Transcription completed for {file_name}")
            logger.info(f"Files saved in {file_dir}")

            # Delete FLAC file if requested
            if args.delete_flac and file_path.endswith('.flac'):
                try:
                    os.remove(file_path)
                    logger.info(f"Deleted temporary FLAC file: {file_path}")
                except Exception as e:
                    logger.error(f"Error deleting FLAC file: {e}")

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