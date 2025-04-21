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
MAX_DAVINCI_WORDS = 500  # maximum words per subtitle block for davinci mode
FILLER_WORDS = ["äh", "ähm"]  # List of filler words to potentially remove
#fill = {'Also,': '', 'dann': '', 'Äh': '', 'ja,': '', 'so': '', 'oder': '', 'weil,': '', 'einfach,': '', 'genau': '', 'quasi': '', 'ähm,': '', 'dann,': '', 'eigentlich,': '', 'weil': '', 'so,': '', 'Ähm,': '', 'eben': '', 'glaube': '', 'Ja,': '', 'mal': '', 'Genau': '', 'eben,': '', 'irgendwie': '', 'mal,': '', 'äh,': '', 'glaube,': '', 'ja': '', 'einfach': '', 'halt': '', 'So': ''}
def setup_logger():
    """Configure loguru logger"""
    logger.remove()  # Remove default handler
    
    # Add file logging
    logger.add(
        "transcribe_{time}.log",
        rotation="1 day",
        retention="7 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    
    # Configure console logging based on verbosity
    if args.debug:
        logger.add(lambda msg: print(msg), level="DEBUG")
    elif args.verbose:
        logger.add(lambda msg: print(msg), level="INFO")
    else:
        logger.add(lambda msg: print(msg), level="ERROR")

def in_debug_mode():
    if args.debug:
        return True
    return False

def clear_screen():
    return

def check_transcript_exists(file_path, file_name):
    json_path = os.path.join(file_path, f"{file_name}.json")
    return os.path.exists(json_path)

def check_audio_length(file_path):
    """Check if audio file is shorter than MAX_AUDIO_LENGTH"""
    audio = AudioSegment.from_file(file_path)
    duration_seconds = len(audio) / 1000.0  # Convert milliseconds to seconds
    if duration_seconds > MAX_AUDIO_LENGTH:
        raise RuntimeError(f"Audio duration ({duration_seconds:.1f}s) exceeds maximum allowed length ({MAX_AUDIO_LENGTH}s)")
    return True

def check_audio_format(audio):
    """Check if audio meets requirements (mono, 16kHz, 16-bit)"""
    return (audio.channels == 1 and 
            audio.frame_rate == 16000 and 
            audio.sample_width == 2)

def convert_to_flac(input_file):
    """Convert audio/video file to FLAC format (mono, 16-bit, 16kHz)"""
    logger.info(f"Converting {input_file} to FLAC format...")
    audio = AudioSegment.from_file(input_file)
    # Convert to mono
    audio = audio.set_channels(1)
    # Set sample rate to 16kHz
    audio = audio.set_frame_rate(16000)
    # Set sample width to 2 bytes (16-bit)
    audio = audio.set_sample_width(2)
    
    # Create output filename
    output_file = os.path.splitext(input_file)[0] + "_converted.flac"
    # Export as FLAC
    audio.export(output_file, format="flac")
    logger.info(f"FLAC conversion completed: {output_file}")
    return output_file

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
    output_file = os.path.splitext(input_file)[0] + "_converted.wav"
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
    
    # Use davinci algorithm if specified
    if args.davinci_srt:
        create_davinci_srt(words, output_file)
        return
        
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

def create_davinci_srt(words, output_file):
    """Create SRT file optimized for Davinci Resolve Studio"""
    logger.info(f"Creating Davinci Resolve optimized SRT file: {output_file}")
    
    # Default pause detection is 200ms if not specified
    pause_detection = args.silentportions if args.silentportions > 0 else 200
    padding = args.padding if hasattr(args, 'padding') else 30
    
    # Always pre-process words to identify filler words, regardless of --remove-fillers flag
    words = process_filler_words(words, pause_detection)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        counter = 1
        block_words = []
        block_start = None
        block_end = None
        
        # Handle initial silence
        if words and words[0]['type'] == 'word' and words[0]['start'] > 0:
            f.write(f"{counter}\n")
            f.write(f"00:00:00,000 --> {format_time(words[0]['start'])}\n")
            f.write("(...)\n\n")
            counter += 1
        
        i = 0
        while i < len(words):
            word = words[i]
            
            # If we find a spacing that exceeds our pause threshold
            if word['type'] == 'spacing':
                duration_ms = (word['end'] - word['start']) * 1000
                if duration_ms >= pause_detection:
                    # Process the accumulated block of words
                    if block_words:
                        process_davinci_block(f, counter, block_words, block_start, block_end)
                        counter += 1
                        block_words = []
                        block_start = None
                        block_end = None
                    
                    # Write silent portion
                    f.write(f"{counter}\n")
                    f.write(f"{format_time(word['start'])} --> {format_time(word['end'])}\n")
                    f.write("(...)\n\n")
                    counter += 1
                else:
                    # Add padding to the previous word if it exists
                    if block_words:
                        block_words[-1]['end'] += padding / 1000.0
                        block_end = block_words[-1]['end']
            
            # Add word to current block
            elif word['type'] == 'word':
                if block_start is None:
                    block_start = word['start']
                block_end = word['end']
                block_words.append(word)
                
                # If we've reached the maximum words limit, process the block
                if len(block_words) >= MAX_DAVINCI_WORDS:
                    process_davinci_block(f, counter, block_words, block_start, block_end)
                    counter += 1
                    block_words = []
                    block_start = None
                    block_end = None
            
            i += 1
        
        # Process any remaining words
        if block_words:
            process_davinci_block(f, counter, block_words, block_start, block_end)
    
    # Post-process the SRT file to merge consecutive pause entries
    merge_consecutive_pauses(output_file)
    
    logger.info("Davinci Resolve SRT file created successfully")

def process_filler_words(words, pause_threshold):
    """Process words to identify and mark filler words as pauses"""
    import re
    logger.info("Processing filler words...")
    
    # Create regex patterns for each filler word (case insensitive, ignoring punctuation)
    filler_patterns = [re.compile(f"^{re.escape(word)}[,.!?]*$", re.IGNORECASE) for word in FILLER_WORDS]
    
    # New list to store processed words
    processed_words = []
    i = 0
    
    while i < len(words):
        # Check if current item is a word and might be a filler
        if words[i]['type'] == 'word':
            is_filler = any(pattern.match(words[i]['text']) for pattern in filler_patterns)
            
            if is_filler and i > 0 and i < len(words) - 1:
                # Check if surrounded by spacings
                prev_spacing = i > 0 and words[i-1]['type'] == 'spacing'
                next_spacing = i < len(words) - 1 and words[i+1]['type'] == 'spacing'
                
                if prev_spacing and next_spacing:
                    # Create a new spacing element replacing previous spacing + filler + next spacing
                    filler_pause = {
                        'type': 'spacing',
                        'start': words[i-1]['start'],
                        'end': words[i+1]['end'],
                        'text': ' ',
                        'speaker_id': words[i].get('speaker_id', 'Unknown')
                    }
                    
                    # Apply padding to the preceding word if it exists
                    if len(processed_words) > 0 and processed_words[-1]['type'] == 'word':
                        # Record the original end time
                        orig_end = processed_words[-1]['end']
                        # Add padding (convert to seconds)
                        if args.padding > 0:
                            processed_words[-1]['end'] += args.padding / 1000.0
                        
                        logger.debug(f"Added {args.padding}ms padding to word ending at {orig_end}")
                    
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
    
    logger.info(f"Filler word processing complete. Removed {len(words) - len(processed_words)} elements")
    return processed_words

def process_davinci_block(f, counter, block_words, start_time, end_time):
    """Process a block of words for Davinci Resolve SRT format"""
    if not block_words:
        return
    
    # Check if we need to split the block
    if len(block_words) > MAX_DAVINCI_WORDS:
        # Find sentence boundaries to split at
        sentences = []
        current_sentence = []
        sentence_end_markers = ['.', '!', '?', ':', ';']
        
        for word in block_words:
            current_sentence.append(word)
            if word['text'] and word['text'][-1] in sentence_end_markers:
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
                
            sentence_text = " ".join([w['text'] for w in sentence])
            sentence_end = sentence[-1]['end']
            
            f.write(f"{current_counter}\n")
            f.write(f"{format_time(current_start)} --> {format_time(sentence_end)}\n")
            f.write(f"{sentence_text}\n\n")
            
            current_start = sentence_end
            current_counter += 1
    else:
        # Write the entire block
        block_text = " ".join([w['text'] for w in block_words])
        f.write(f"{counter}\n")
        f.write(f"{format_time(start_time)} --> {format_time(end_time)}\n")
        f.write(f"{block_text}\n\n")

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

def merge_consecutive_pauses(srt_file):
    """Post-process SRT file to merge consecutive pause entries"""
    import re
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
        
        # If no merge happened, add the current entry as is
        merged_entries.append(entries[i])
        i += 1
    
    # Write back the merged content
    with open(srt_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(merged_entries))
    
    logger.info(f"Merged {len(entries) - len(merged_entries)} consecutive pause entries")

def min_timestamp(ts1, ts2):
    """Return the earlier of two timestamps in SRT format (HH:MM:SS,mmm)"""
    h1, m1, rest1 = ts1.split(':')
    s1, ms1 = rest1.split(',')
    
    h2, m2, rest2 = ts2.split(':')
    s2, ms2 = rest2.split(',')
    
    time1 = int(h1) * 3600 + int(m1) * 60 + int(s1) + int(ms1) / 1000.0
    time2 = int(h2) * 3600 + int(m2) * 60 + int(s2) + int(ms2) / 1000.0
    
    return ts1 if time1 <= time2 else ts2

def max_timestamp(ts1, ts2):
    """Return the later of two timestamps in SRT format (HH:MM:SS,mmm)"""
    h1, m1, rest1 = ts1.split(':')
    s1, ms1 = rest1.split(',')
    
    h2, m2, rest2 = ts2.split(':')
    s2, ms2 = rest2.split(',')
    
    time1 = int(h1) * 3600 + int(m1) * 60 + int(s1) + int(ms1) / 1000.0
    time2 = int(h2) * 3600 + int(m2) * 60 + int(s2) + int(ms2) / 1000.0
    
    return ts1 if time1 >= time2 else ts2

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
    parser.add_argument("--use-pcm", help="Use PCM format instead of FLAC (larger file size)", action="store_true")
    parser.add_argument("-l", "--language", help="Language code (ISO-639-1 or ISO-639-3). Examples: en (English), fr (French), de (German)", default=None)
    parser.add_argument("-v", "--verbose", help="Show all log messages in console", action="store_true")
    parser.add_argument("--force", help="Force re-transcription even if files exist", action="store_true")
    parser.add_argument("-p", "--silentportions", type=int, help="Mark pauses longer than X milliseconds with (...)", default=0)
    parser.add_argument("--davinci-srt", "-D", help="Export SRT for Davinci Resolve with optimized subtitle blocks", action="store_true")
    parser.add_argument("--padding", type=int, help="Add X milliseconds padding to word end times (default: 30ms)", default=30)
    parser.add_argument("--remove-fillers", help="Remove filler words like 'äh' and 'ähm' and treat them as pauses", action="store_true")
    
    args = parser.parse_args()
    pp = pprint.PrettyPrinter(indent=4)

    # Setup logging
    setup_logger()
    if args.debug:
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

        # Convert to appropriate format if needed
        if not args.no_convert:
            if file_extension.lower() == '.wav':
                # Check if WAV needs re-encoding
                audio = AudioSegment.from_file(file_path)
                if not check_audio_format(audio):
                    logger.info("WAV file needs re-encoding to meet requirements")
                    file_path = convert_to_pcm(file_path) if args.use_pcm else convert_to_flac(file_path)
            elif file_extension.lower() == '.flac' and not args.use_pcm:
                # Check if FLAC needs re-encoding
                audio = AudioSegment.from_file(file_path)
                if not check_audio_format(audio):
                    logger.info("FLAC file needs re-encoding to meet requirements")
                    file_path = convert_to_flac(file_path)
            else:
                # Convert other formats
                file_path = convert_to_pcm(file_path) if args.use_pcm else convert_to_flac(file_path)

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
                    'file': ('audio.wav' if args.use_pcm else 'audio.flac', audio_file, 'audio/wav' if args.use_pcm else 'audio/flac')
                }
                data = {
                    'model_id': 'scribe_v1',
                    'language_code': args.language,
                    'tag_audio_events': 'true',
                    'num_speakers': '2' if args.speaker_labels else None,
                    'timestamps_granularity': 'word',
                    'diarize': 'true' if args.speaker_labels else 'false',
                    'file_format': 'pcm_s16le_16' if args.use_pcm else 'other'
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

            # Delete temporary file by default unless --keep-flac is specified
            if not args.keep_flac and (file_path.endswith('_converted.wav') or file_path.endswith('_converted.flac')):
                try:
                    # Add a small delay to ensure file is released by OS
                    time.sleep(0.5)
                    os.remove(file_path)
                    logger.info(f"Deleted temporary file: {file_path}")
                except Exception as e:
                    logger.error(f"Error deleting temporary file: {e}")

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