# #!/usr/bin/env python3
# /// script
# dependencies = [
#   "load_dotenv",
#   "argparse",
#   "requests",
#   "datetime",
#   "pydub",
#   "assemblyai",
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
import json
import argparse

#from openai import OpenAI
#import pydub
from pydub import AudioSegment
# Docs https://github.com/jiaaro/pydub/blob/master/API.markdown

import assemblyai as aai
from transcribe_helpers import create_word_level_srt
from transcribe_helpers.text_processing import process_filler_words, standardize_word_format

# global variables
args = ""
headers = ""

# Function to check if we are in debug mode
def in_debug_mode():
    #if len(sys.argv) > 1 and sys.argv[1] == "--debug":
    if args.debug:
        # debug = True
        return True
    # gettrace = getattr(sys, 'gettrace', None)
    # if gettrace is not None:
    #     # debug = True
    #     print( getattr(sys, 'gettrace', None))
    # => <built-in function gettrace> refactor this!
    # return True

def in_verbose_mode():
    return args.verbose

def clear_screen():
    # os.system('cls' if os.name == 'nt' else 'clear')
    return

def check_transcript_exists(file_path, file_name):
    transcript_path = os.path.join(file_path, f"{file_name}.txt")
    srt_path = os.path.join(file_path, f"{file_name}.srt")
    return os.path.exists(transcript_path) or os.path.exists(srt_path)

def check_json_exists(file_path, file_name):
    """Check if a JSON transcript file exists for a given file name."""
    # Check for API-specific JSON
    json_path = os.path.join(file_path, f"{file_name}_assemblyai.json")
    if os.path.exists(json_path):
        logger.info(f"Found AssemblyAI JSON file: {json_path}")
        return True, json_path
    
    # Check for generic JSON as fallback
    json_path = os.path.join(file_path, f"{file_name}.json")
    if os.path.exists(json_path):
        logger.info(f"Found generic JSON file: {json_path}")
        return True, json_path
        
    return False, ""

def load_json_transcript(file_path, file_name):
    """Load transcript data from a saved JSON file."""
    json_path = os.path.join(file_path, f"{file_name}.json")
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Error loading JSON file: {e}")
        return None

def export_subtitles(transcript_id, subtitle_format, file_name):
    """
    Export subtitles using AssemblyAI API.

    Args:
        api_token (str): Your API token for AssemblyAI.
        transcript_id (str): The ID of the transcript you want to export as subtitles.
        subtitle_format (str): The subtitle format to export (either 'srt' or 'vtt').

    Returns:
        str: The filename of the saved subtitle file.
    """
    # Import from transcribe_helpers
    from transcribe_helpers.output_formatters import export_subtitles as th_export_subtitles
    from transcribe_helpers.output_formatters import custom_export_subtitles
    
    # Use custom_export_subtitles if we need pause indicators or specific formatting
    if subtitle_format == 'srt' and (args.silentportions > 0 or args.show_pauses):
        return custom_export_subtitles(
            transcript_id, 
            headers, 
            file_name,
            show_pauses=args.show_pauses,
            silentportions=args.silentportions or 250,
            chars_per_line=args.chars_per_line,
            padding_start=args.padding_start,
            padding_end=args.padding_end,
            fps=args.fps, 
            fps_offset_start=args.fps_offset_start, 
            fps_offset_end=args.fps_offset_end,
            remove_fillers=args.remove_fillers and not args.no_remove_fillers,
            filler_words=["äh", "ähm", "uh", "um", "ah", "er", "hm", "hmm"]
        )
    else:
        # Use standard export for other formats or when no custom formatting needed
        return th_export_subtitles(transcript_id, headers, subtitle_format, file_name, 
                                 args.fps, args.fps_offset_start, args.fps_offset_end)

def main():
    global args, headers
    parser = argparse.ArgumentParser(description="Audio transcription using AssemblyAI API.")
    
    # Positional argument for audio path
    parser.add_argument("audio_path", help="Path to the input audio file or pattern.")
    
    # Debug and verbose options
    parser.add_argument("-d", "--debug", help="Debug mode", action="store_true")
    parser.add_argument("-v", "--verbose", help="Show all log messages in console", action="store_true")
    
    # Transcription control
    parser.add_argument("--force", help="Force re-transcription even if files exist", action="store_true")
    
    # File handling options
    file_group = parser.add_mutually_exclusive_group()
    file_group.add_argument("--no-convert", help="[DEPRECATED] Use --use-input instead", action="store_true")
    file_group.add_argument("--use-pcm", help="Convert to PCM WAV format (larger file size)", action="store_true")
    file_group.add_argument("--use-input", help="Use original input file without conversion (default is to convert to FLAC)", action="store_true")
    parser.add_argument("--keep-flac", help="Keep the generated FLAC file after processing", action="store_true")
    
    # Language options
    parser.add_argument("-l", "--language", help="Language code (ISO-639-1 or ISO-639-3)", default="en_us")
    parser.add_argument("--language-detection", help="Enable automatic language detection", action="store_true")
    
    # Text formatting options
    parser.add_argument("-c", "--chars-per-line", type=int, help="Maximum characters per line in SRT file (default: 80). Ignored if -C/--word-srt is set.", default=80)
    parser.add_argument("-C", "--word-srt", action="store_true", help="Output SRT with each word as its own subtitle (word-level SRT, disables -c/--chars-per-line)")
    parser.add_argument("-D", "--davinci-srt", action="store_true", help="Export SRT for Davinci Resolve with optimized subtitle blocks (sets: chars-per-line=500, silentportions=250ms, padding-start=-125ms, remove_fillers=True, max 500 words/block). Can override with additional args.")
    
    # Timing options
    parser.add_argument("-p", "--silentportions", type=int, help="Mark pauses longer than X milliseconds with (...) (with -D: default 250ms)", default=0)
    parser.add_argument("--padding-start", type=int, help="Milliseconds to offset word start times into preceding silence (negative=earlier, positive=later, default: 0ms, with -D: -125ms)", default=0)
    parser.add_argument("--padding-end", type=int, help="Milliseconds to offset word end times into following silence (negative=earlier, positive=later, default: 0ms)", default=0)
    parser.add_argument("--padding", type=int, help="DEPRECATED: Use --padding-end instead", default=None)
    parser.add_argument("--show-pauses", help="Add (...) text for pauses longer than silentportions value (auto-enabled with -D)", action="store_true")
    
    # Processing options
    remove_fillers_group = parser.add_mutually_exclusive_group()
    remove_fillers_group.add_argument("--remove-fillers", help="Remove filler words, audio events, and text in parentheses (auto-enabled with -D)", action="store_true")
    remove_fillers_group.add_argument("--no-remove-fillers", help="Do not remove filler words (can override -D default)", action="store_true")
    
    # Speaker and timing options
    parser.add_argument("-s", "--speaker_labels", help="Use this flag to remove speaker labels", action="store_false", default=True)
    parser.add_argument("--fps", type=float, help="Frames per second for frame-based editing (e.g., 24, 29.97, 30)")
    parser.add_argument("--fps-offset-start", type=int, help="Frames to offset from start time (default: -1, negative=earlier, positive=later)", default=-1)
    parser.add_argument("--fps-offset-end", type=int, help="Frames to offset from end time (default: 0, negative=earlier, positive=later)", default=0)
    
    # Legacy/compatibility options
    parser.add_argument("-i", "--id", help=argparse.SUPPRESS)  # Hidden option for ID compatibility
    
    args = parser.parse_args()

    # Handle davinci-srt defaults
    if args.davinci_srt:
        if args.chars_per_line == 80:  # Only set if user didn't specify
            args.chars_per_line = 500
        if args.silentportions == 0:   # Only set if user didn't specify
            args.silentportions = 250
        if args.padding_start == 0:    # Only set if user didn't specify
            args.padding_start = -125
        if not args.no_remove_fillers:  # Only set if user didn't explicitly disable
            args.remove_fillers = True
        if not args.show_pauses:       # Only set if user didn't explicitly specify
            args.show_pauses = True

    # Support deprecated padding parameter
    if args.padding is not None:
        args.padding_end = args.padding

    pp = pprint.PrettyPrinter(indent=4)

    if in_debug_mode() or in_verbose_mode():
        clear_screen()
        print("Debug mode:" if in_debug_mode() else "Verbose mode:" + " active") 

    load_dotenv() # load environment variables from .env file
    aai.settings.api_key = os.getenv("ASSEMBLY_AI_KEY")
    # For direct requests calls
    headers = {
                "Authorization": os.getenv("ASSEMBLY_AI_KEY")
            }

    # Initialize an empty dictionary to store the files
    files_dict = {}

    # Check if args.id is given
    if args.id:
        # If id is given, skip this part and make a dummy filename entry with CWD/{id}.wav
        files_dict[args.id] = os.path.join(os.getcwd(), f"{args.id}.wav")
    else:
        # Process the audio_path argument
        audio_path = args.audio_path
        
        # Check if audio_path is a directory
        if os.path.isdir(audio_path):
            if in_debug_mode() or in_verbose_mode():
                print("Directory found.")
            # If it's a directory, get all audio and video files in the directory
            for root, dirs, files in os.walk(audio_path):
                for file in files:
                    if file.endswith(('.mp3', '.wav', '.ogg', '.mp4', '.avi', '.mp4', '.mov', '.mkv', '.webm', '.m4a', '.flac', '.aac', '.wma', '.aiff', '.flv', '.wmv', '.3gp', '.3g2', '.m4v', '.ts', '.m2ts', '.mts', '.vob', '.ogv', '.ogg', '.oga', '.opus', '.spx', '.amr', '.mka', '.mk3d')):
                        files_dict[file] = os.path.join(root, file)
        # Check if audio_path is a file
        elif os.path.isfile(audio_path):
            # If it's a file, remove any leading './' or '.\' and add it to the dictionary
            normalized_file = os.path.normpath(audio_path)
            files_dict[normalized_file] = normalized_file
            if in_debug_mode() or in_verbose_mode():
                print("File found.")
        # Check if audio_path is a wildcard pattern
        elif '*' in audio_path or '?' in audio_path:
            if in_debug_mode() or in_verbose_mode():
                print("Wildcard pattern found.")
            # If it's a wildcard pattern, find all matching files
            for file in glob.glob(audio_path):
                files_dict[file] = file
        else:
            print("Invalid input. Please provide a valid file, directory, or wildcard pattern.")
            return

    # If no files were found and no ID provided, list available transcripts
    if not files_dict and not args.id:
        print("Listing transcripts:")
        # curl https://api.assemblyai.com/v2/transcript -H "Authorization: <apiKey>"

        url = "https://api.assemblyai.com/v2/transcript"

        response = requests.get(url, headers=headers)

        data = response.json()

        # Create a list to store the transcript data
        transcript_data = []

        print("Number | ID                                   | Created             | Error")
        print("-" * 80)

        for i, transcript in enumerate(data["transcripts"]):
            if i >= 5:
                break
            created = datetime.strptime(transcript["created"], "%Y-%m-%dT%H:%M:%S.%f")
            error = transcript.get("error", "")
            transcript_data.append((transcript['id'], i+1))
            print(f"{i+1:6} | {transcript['id']} | {created.strftime('%Y-%m-%d %H:%M:%S')} | {error}")

        # Ask the user for input
        selected_number = input("Enter the number of the transcript you want to select (or press Enter to exit): ")

        # Save the response
        response = None
        if selected_number:
            selected_number = int(selected_number)
            if selected_number > 0 and selected_number <= len(transcript_data):
                args.id = transcript_data[selected_number-1][0]

        if not args.id:
            print("No transcript selected. Exiting...")    
            exit()
    
    for file_name, file_path in files_dict.items():
        if in_debug_mode() or in_verbose_mode():
            print(f"File: {file_name}")
        # check if file exists
        if not os.path.exists(file_path) and not args.id:
            print(f"Audio File {file_name} does not exist!")
            continue

        full_file_name = os.path.basename(file_path)
        file_name_without_ext, file_extension = os.path.splitext(full_file_name)
        # Use the input filename if requested
        if args.use_input:
            file_name = file_name_without_ext
        else:    
            file_name = file_name_without_ext

        # cd to file_path
        if(os.path.dirname(file_path) != ""):
            os.chdir(os.path.dirname(file_path))
        # save dir_name
        file_dir = os.path.dirname(file_path)

        # Check if transcript exists and force is not enabled
        if check_transcript_exists(os.path.dirname(file_path), file_name) and not args.force:
            print(f"Transcript for {file_name} exists!")
            continue

        # Initialize transcript_json to None
        transcript_json = None
        transcript_id = None
        
        # Check for existing JSON (unless --force is used)
        json_exists, json_path = check_json_exists(file_dir, file_name)
        if json_exists and not args.force:
            if in_debug_mode() or in_verbose_mode():
                print(f"Loading existing transcript JSON for {file_name}")
            
            # Load existing JSON file
            with open(json_path, 'r') as f:
                transcript_json = json.load(f)
            if transcript_json:
                transcript_id = transcript_json.get('id')
                print(f"Using existing transcript ID: {transcript_id}")
                
                # Create a dummy transcript object with the ID
                class Transcript:
                    def __init__(self, id):
                        self.id = id
                transcript = Transcript(transcript_id)
                
                # Generate output files from existing JSON
                if in_debug_mode() or in_verbose_mode():
                    print("Getting Sentences...")    
                
                # Then get the sentences
                sentences_url = f"https://api.assemblyai.com/v2/transcript/{transcript.id}/sentences"
                sentences_response = requests.get(sentences_url, headers=headers)
                    
                # Get sentences and create text output
                with open(os.path.join(file_dir, f"{file_name}.txt"), "w") as f:
                    current_speaker = ""
                    # Iterate over the "sentences" list
                    for sentence in json.loads(sentences_response.text)['sentences']:
                        # Extract the "text" field and the speaker
                        text = sentence['text']
                        if args.speaker_labels:
                            speaker = sentence['speaker']
                            if speaker != current_speaker:
                                # Speaker change
                                current_speaker = speaker
                                f.write(f"Speaker {speaker}: {text}\n")
                            else:
                                # same speaker
                                f.write(f"{text}\n")
                        else:
                            f.write(f"{text}\n")
                
                # Handle SRT generation using existing transcript ID
                if args.word_srt or args.davinci_srt:
                    # Get words from the transcript to create word-level SRT
                    words_url = f"https://api.assemblyai.com/v2/transcript/{transcript.id}/words"
                    words_response = requests.get(words_url, headers=headers)
                    
                    if words_response.status_code == 200:
                        # Process with standardized word format
                        words_data = json.loads(words_response.text)
                        processed_words = standardize_word_format(
                            words_data.get('words', []),
                            'assemblyai',
                            show_pauses=args.show_pauses,
                            silence_threshold=args.silentportions or 250
                        )
                        
                        # Process filler words if needed
                        if args.remove_fillers and not args.no_remove_fillers:
                            filler_words = ["äh", "ähm", "uh", "um", "ah", "er", "hm", "hmm"]
                            processed_words = process_filler_words(processed_words, args.silentportions or 250, filler_words)
                        
                        # Create appropriate SRT based on mode
                        srt_file = os.path.join(file_dir, f"{file_name}.srt")
                        
                        if args.davinci_srt:
                            create_davinci_srt(
                                processed_words,
                                srt_file,
                                silentportions=args.silentportions or 250,
                                padding_start=args.padding_start,
                                padding_end=args.padding_end,
                                fps=args.fps,
                                fps_offset_start=args.fps_offset_start,
                                fps_offset_end=args.fps_offset_end,
                                remove_fillers=False  # Already handled
                            )
                            print(f"Davinci Resolve optimized SRT saved to {file_name}.srt")
                        else:
                            create_word_level_srt(
                                processed_words, 
                                srt_file, 
                                remove_fillers=False,  # Already handled
                                fps=args.fps,
                                fps_offset_start=args.fps_offset_start,
                                fps_offset_end=args.fps_offset_end,
                                padding_start=args.padding_start, 
                                padding_end=args.padding_end
                            )
                            print(f"Word-level SRT saved to {file_name}.srt")
                    else:
                        # Fallback to standard SRT export if words endpoint fails
                        print(f"Failed to get word-level data, using standard SRT export")
                        export_subtitles(transcript.id, "srt", file_name)
                else:
                    # Standard SRT export
                    export_subtitles(transcript.id, "srt", file_name)
                
                print(f"Regenerated files for {file_name} from existing JSON")
                continue
            else:
                print(f"Failed to load JSON file, will request new transcription")
        
        # If we don't have valid JSON and transcript ID, request a new transcription
        if not transcript_json or args.force:
            if not args.id:
                # https://platform.openai.com/docs/guides/speech-to-text/longer-inputs
                # G:\Geteilte Ablagen\_3_References\Transscribe Queue\v12044gd0000cg3j4tbc77ubn6e7us3g.mp4
                audio = AudioSegment.from_file(file_path)

                # What are the API limits on file size or file duration?
                # Currently, the maximum file size that can be submitted to the /v2/transcript endpoint for transcription is 5GB, and the maximum duration is 10 hours.
                # The maximum file size for a local file uploaded to the API via the /v2/upload endpoint is 2.2GB.

                # raw_date byte length
                if in_debug_mode() or in_verbose_mode():
                    mb = len(audio.raw_data) / (1024.0 * 1024.0)
                    print(f"Audio size: {mb} MB")
                
                # https://platform.openai.com/docs/guides/speech-to-text/longer-inputs
                # PyDub handles time in milliseconds
                twenty_four_minutes = 24 * 60 * 1000

                # Create configuration with language from command line
                config = aai.TranscriptionConfig(
                    speaker_labels=args.speaker_labels, 
                    format_text=True, 
                    language_code=args.language,
                    language_detection=args.language_detection,
                    speech_model="best"
                )
                # https://www.assemblyai.com/docs/api-reference/transcripts/submit
                # speakers_expected
                # language_code
                # speech_model
                # auto_highlights https://www.assemblyai.com/docs/audio-intelligence/key-phrases
                # audio_start_from
                # audio_end_at

                if in_debug_mode() or in_verbose_mode():
                    print(f"Starting AssemblyAI Transcription with language '{args.language}'...")
                    
                # id - '030df9ec-d65c-4b30-bab5-7e00fe56a9de'

                if not args.id:
                    transcriber = aai.Transcriber()
                    transcript = transcriber.transcribe(
                        file_path,
                        config=config
                        )
                    
                    print(f"Transcript ID: {transcript.id}")
                    
                    # this is a dict
                    transcript_json = transcript.json_response
                    # json dict to json
                    json_str = json.dumps(transcript_json)
                    
                    print(f"Transcription Length: {transcript_json['audio_duration']}")
                    
                    # transcript.json_response['text']
                    # transcript.text

                    # Save the transcript object as a json file
                    with open(os.path.join(file_dir, f"{file_name}_assemblyai.json"), "w") as f:
                        f.write(json.dumps(transcript_json))
                else:    
                    #aai.Transcript(args.id)
                    class Transcript:
                        def __init__(self, id):
                            self.id = id
                    transcript = Transcript(args.id)
                    
                    # Get full transcript json from API if using an ID
                    transcript_url = f"https://api.assemblyai.com/v2/transcript/{transcript.id}"
                    transcript_response = requests.get(transcript_url, headers=headers)
                    
                    if transcript_response.status_code == 200:
                        transcript_json = json.loads(transcript_response.text)
                        # Save the transcript object as a json file
                        with open(os.path.join(file_dir, f"{file_name}_assemblyai.json"), "w") as f:
                            f.write(json.dumps(transcript_json))
            else:
                transcript = Transcript(args.id)
                # Get full transcript json from API if using an ID
                transcript_url = f"https://api.assemblyai.com/v2/transcript/{transcript.id}"
                transcript_response = requests.get(transcript_url, headers=headers)
                
                if transcript_response.status_code == 200:
                    transcript_json = json.loads(transcript_response.text)
                    # Save the transcript object as a json file
                    with open(os.path.join(file_dir, f"{file_name}_assemblyai.json"), "w") as f:
                        f.write(json.dumps(transcript_json))

        if in_debug_mode() or in_verbose_mode():
            print("Getting Sentences...")    
        
        # Now generate the output files regardless of whether we used an existing JSON or created a new one
        
        # Then get the sentences
        sentences_url = f"https://api.assemblyai.com/v2/transcript/{transcript.id}/sentences"
        sentences_response = requests.get(sentences_url, headers=headers)
            
        # Get sentences and create text output
        with open(os.path.join(file_dir, f"{file_name}.txt"), "w") as f:
            current_speaker = ""
            # Iterate over the "sentences" list
            for sentence in json.loads(sentences_response.text)['sentences']:
                # Extract the "text" field and the speaker
                text = sentence['text']
                if args.speaker_labels:
                    speaker = sentence['speaker']
                    if speaker != current_speaker:
                        # Speaker change
                        current_speaker = speaker
                        f.write(f"Speaker {speaker}: {text}\n")
                    else:
                        # same speaker
                        f.write(f"{text}\n")
                else:
                    f.write(f"{text}\n")
        
        # Create SRT files
        if args.word_srt or args.davinci_srt:
            # Get words from the transcript to create word-level SRT
            words_url = f"https://api.assemblyai.com/v2/transcript/{transcript.id}/words"
            words_response = requests.get(words_url, headers=headers)
            
            if words_response.status_code == 200:
                words_data = json.loads(words_response.text)
                
                # Use standardize_word_format to convert AssemblyAI format to our standard format
                processed_words = standardize_word_format(
                    words_data.get('words', []),
                    'assemblyai',
                    show_pauses=args.show_pauses,
                    silence_threshold=args.silentportions or 250
                )
                
                # Process filler words if needed
                if args.remove_fillers and not args.no_remove_fillers:
                    filler_words = ["äh", "ähm", "uh", "um", "ah", "er", "hm", "hmm"]
                    processed_words = process_filler_words(processed_words, args.silentportions or 250, filler_words)
                
                # Convert frame offsets to milliseconds if FPS is specified
                start_offset_ms = args.padding_start
                end_offset_ms = args.padding_end
                
                if args.fps and args.fps_offset_start != 0:
                    # Calculate frame offset in milliseconds
                    frame_ms = 1000.0 / args.fps
                    start_offset_ms += int(args.fps_offset_start * frame_ms)
                
                if args.fps and args.fps_offset_end != 0:
                    # Calculate frame offset in milliseconds
                    frame_ms = 1000.0 / args.fps
                    end_offset_ms += int(args.fps_offset_end * frame_ms)
                
                # Create appropriate SRT based on mode
                srt_file = os.path.join(file_dir, f"{file_name}.srt")
                
                if args.davinci_srt:
                    # Import directly here to avoid potential circular imports
                    from transcribe_helpers.output_formatters import create_davinci_srt
                    create_davinci_srt(
                        processed_words,
                        srt_file,
                        silentportions=args.silentportions or 250,
                        padding_start=start_offset_ms,
                        padding_end=end_offset_ms,
                        fps=args.fps,
                        fps_offset_start=args.fps_offset_start,
                        fps_offset_end=args.fps_offset_end,
                        remove_fillers=False  # Already handled
                    )
                    print(f"Davinci Resolve optimized SRT saved to {file_name}.srt")
                else:
                    # Word-level SRT
                    create_word_level_srt(
                        processed_words, 
                        srt_file, 
                        remove_fillers=False,  # Already handled by process_filler_words
                        fps=args.fps,
                        fps_offset_start=args.fps_offset_start,
                        fps_offset_end=args.fps_offset_end,
                        padding_start=start_offset_ms, 
                        padding_end=end_offset_ms
                    )
                    print(f"Word-level SRT saved to {file_name}.srt")
            else:
                print(f"Failed to get word-level data: {words_response.status_code} - {words_response.text}")
                # Try to get the entire transcript data instead
                transcript_url = f"https://api.assemblyai.com/v2/transcript/{transcript.id}"
                transcript_response = requests.get(transcript_url, headers=headers)
                
                if transcript_response.status_code == 200:
                    transcript_data = json.loads(transcript_response.text)
                    if 'words' in transcript_data and transcript_data['words']:
                        # Use standardize_word_format for transcript data words too
                        processed_words = standardize_word_format(
                            transcript_data['words'],
                            'assemblyai',
                            show_pauses=args.show_pauses,
                            silence_threshold=args.silentportions or 250
                        )
                        
                        # Process filler words if needed
                        if args.remove_fillers and not args.no_remove_fillers:
                            filler_words = ["äh", "ähm", "uh", "um", "ah", "er", "hm", "hmm"]
                            processed_words = process_filler_words(processed_words, args.silentportions or 250, filler_words)
                        
                        # Convert frame offsets to milliseconds if FPS is specified
                        start_offset_ms = args.padding_start
                        end_offset_ms = args.padding_end
                        
                        if args.fps and args.fps_offset_start != 0:
                            # Calculate frame offset in milliseconds
                            frame_ms = 1000.0 / args.fps
                            start_offset_ms += int(args.fps_offset_start * frame_ms)
                        
                        if args.fps and args.fps_offset_end != 0:
                            # Calculate frame offset in milliseconds
                            frame_ms = 1000.0 / args.fps
                            end_offset_ms += int(args.fps_offset_end * frame_ms)

                        # Create appropriate SRT based on mode
                        srt_file = os.path.join(file_dir, f"{file_name}.srt")
                        
                        if args.davinci_srt:
                            # Import directly here to avoid potential circular imports
                            from transcribe_helpers.output_formatters import create_davinci_srt
                            create_davinci_srt(
                                processed_words,
                                srt_file,
                                silentportions=args.silentportions or 250,
                                padding_start=start_offset_ms,
                                padding_end=end_offset_ms,
                                fps=args.fps,
                                fps_offset_start=args.fps_offset_start,
                                fps_offset_end=args.fps_offset_end,
                                remove_fillers=False  # Already handled
                            )
                            print(f"Davinci Resolve optimized SRT saved to {file_name}.srt")
                        else:
                            # Word-level SRT
                            create_word_level_srt(
                                processed_words, 
                                srt_file, 
                                remove_fillers=False,  # Already handled by process_filler_words
                                fps=args.fps,
                                fps_offset_start=args.fps_offset_start,
                                fps_offset_end=args.fps_offset_end,
                                padding_start=start_offset_ms, 
                                padding_end=end_offset_ms
                            )
                            print(f"Word-level SRT saved to {file_name}.srt using transcript data")
                    else:
                        print("No word-level data found in transcript. Using standard SRT export instead.")
                        export_subtitles(transcript.id, "srt", file_name)
                else:
                    print(f"Failed to get transcript data. Using standard SRT export instead.")
                    export_subtitles(transcript.id, "srt", file_name)
        else:
            # Standard SRT export
            export_subtitles(transcript.id, "srt", file_name)
        
        print(f"Transcript saved to {file_name}.txt")
        
if __name__ == '__main__':
    main()        