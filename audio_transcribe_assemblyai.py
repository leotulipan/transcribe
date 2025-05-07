# /// script
# dependencies = [
#   "load_dotenv",
#   "argparse",
#   "requests",
#   "datetime",
#   "assemblyai",
#   "pydub",
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

def clear_screen():
    # os.system('cls' if os.name == 'nt' else 'clear')
    return

def check_transcript_exists(file_path, file_name):
    transcript_path = os.path.join(file_path, f"{file_name}.txt")
    srt_path = os.path.join(file_path, f"{file_name}.srt")
    return os.path.exists(transcript_path) or os.path.exists(srt_path)

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
    # Import from transcribe_helpers if it hasn't been imported yet
    from transcribe_helpers.output_formatters import export_subtitles as th_export_subtitles
    
    # Call the function from transcribe_helpers
    return th_export_subtitles(transcript_id, headers, subtitle_format, file_name, 
                             args.fps, args.fps_offset_start, args.fps_offset_end)

def main():
    global args, headers
    parser = argparse.ArgumentParser()
    # debug 
    parser.add_argument("-d", "--debug", help="Debug mode", action="store_true")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("-f", "--file", "--folder", help="filename, foldername, or pattern to transcribe")
    group.add_argument("-i", "--id", help="ID of an already done transcription")
    # speaker_labels
    parser.add_argument("-s", "--speaker_labels", help="Use this flag to remove speaker labels", action="store_false", default=True) 
    parser.add_argument("-C", "--word-srt", action="store_true", help="Output SRT with each word as its own subtitle (word-level SRT)")
    parser.add_argument("--remove-fillers", help="Remove filler words like 'äh' and 'ähm' and treat them as pauses", action="store_true")
    parser.add_argument("--fps", type=float, help="Frames per second for frame-based editing (e.g., 24, 29.97, 30)", default=None)
    parser.add_argument("--fps-offset-start", type=int, help="Frames to offset from start time (default: 1)", default=1)
    parser.add_argument("--fps-offset-end", type=int, help="Frames to offset from end time (default: 0)", default=0)
    
    args = parser.parse_args()

    pp = pprint.PrettyPrinter(indent=4)

    #client = OpenAI()

    if in_debug_mode():
        clear_screen()
        print("We are in debug mode.") 

    load_dotenv() # load environment variables from .env file
    aai.settings.api_key = os.getenv("ASSEMBLY_AI_KEY")
    # For direct requests calls
    headers = {
                "Authorization": os.getenv("ASSEMBLY_AI_KEY")
            }


    # if no file or id is given, call the list transcript api
    if not args.file and not args.id:
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

        
    # Initialize an empty dictionary to store the files
    files_dict = {}

    # Check if args.id is given
    if args.id:
        # If id is given, skip this part and make a dummy filename entry with CWD/{id}.wav
        files_dict[args.id] = os.path.join(os.getcwd(), f"{args.id}.wav")
    else:
        # Check if args.file is a directory
        if os.path.isdir(args.file):
            if in_debug_mode():
                print("Directory found.")
            # If it's a directory, get all audio and video files in the directory
            for root, dirs, files in os.walk(args.file):
                for file in files:
                    if file.endswith(('.mp3', '.wav', '.ogg', '.mp4', '.avi', '.mp4', '.mov', '.mkv', '.webm', '.m4a', '.flac', '.aac', '.wma', '.aiff', '.flv', '.wmv', '.3gp', '.3g2', '.m4v', '.ts', '.m2ts', '.mts', '.vob', '.ogv', '.ogg', '.oga', '.opus', '.spx', '.amr', '.mka', '.mk3d')):
                        files_dict[file] = os.path.join(root, file)
        # Check if args.file is a file
        elif os.path.isfile(args.file):
            # If it's a file, remove any leading './' or '.\' and add it to the dictionary
            normalized_file = os.path.normpath(args.file)
            files_dict[normalized_file] = normalized_file
            if in_debug_mode():
                print("File found.")
        # Check if args.file is a wildcard pattern
        elif '*' in args.file or '?' in args.file:
            if in_debug_mode():
                print("Wildcard pattern found.")
            # If it's a wildcard pattern, find all matching files
            for file in glob.glob(args.file):
                files_dict[file] = file
        else:
            print("Invalid input. Please provide a valid file, directory, or wildcard pattern.")
            return
    
    for file_name, file_path in files_dict.items():
        if in_debug_mode():
            print(f"File: {file_name}")
        # check if file exists
        if not os.path.exists(file_path) and not args.id:
            print(f"Audio File {file_name} does not exist!")
            continue

        full_file_name = os.path.basename(file_path)
        file_name, file_extension = os.path.splitext(full_file_name)    

        # cd to file_path
        if(os.path.dirname(file_path) != ""):
            os.chdir(os.path.dirname(file_path))
        # save dir_name
        file_dir = os.path.dirname(file_path)

        # Check if transcript exists
        if check_transcript_exists(os.path.dirname(file_path), file_name):
            print(f"Transcript for {file_name} exists!")
            continue

        if not args.id:
            # https://platform.openai.com/docs/guides/speech-to-text/longer-inputs
            # G:\Geteilte Ablagen\_3_References\Transscribe Queue\v12044gd0000cg3j4tbc77ubn6e7us3g.mp4
            audio = AudioSegment.from_file(file_path)

            # What are the API limits on file size or file duration?
            # Currently, the maximum file size that can be submitted to the /v2/transcript endpoint for transcription is 5GB, and the maximum duration is 10 hours.
            # The maximum file size for a local file uploaded to the API via the /v2/upload endpoint is 2.2GB.

            # raw_date byte length
            if in_debug_mode():
                mb = len(audio.raw_data) / (1024.0 * 1024.0)
                print(f"Audio size: {mb} MB")
            
        # https://platform.openai.com/docs/guides/speech-to-text/longer-inputs
        # PyDub handles time in milliseconds
        twenty_four_minutes = 24 * 60 * 1000

        config = aai.TranscriptionConfig(speaker_labels=args.speaker_labels, format_text=True, language_code="de", speech_model="best")
        # https://www.assemblyai.com/docs/api-reference/transcripts/submit
        # speakers_expected
        # language_code="de"
        # speech_model
        # auto_highlights https://www.assemblyai.com/docs/audio-intelligence/key-phrases
        # audio_start_from
        # audio_end_at

        if in_debug_mode():
            print("Starting AssemblyAI Transcription...")
            
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
            with open(os.path.join(file_dir, f"{file_name}.json"), "w") as f:
                f.write(json.dumps(transcript_json))
        else:    
            #aai.Transcript(args.id)
            class Transcript:
                def __init__(self, id):
                    self.id = id
            transcript = Transcript(args.id)

        if in_debug_mode():
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
        
        # Create SRT files
        if args.word_srt:
            # Get words from the transcript to create word-level SRT
            words_url = f"https://api.assemblyai.com/v2/transcript/{transcript.id}/words"
            words_response = requests.get(words_url, headers=headers)
            
            if words_response.status_code == 200:
                words_data = json.loads(words_response.text)
                # Convert AssemblyAI words format to the format expected by create_word_level_srt
                words = []
                for word in words_data.get('words', []):
                    words.append({
                        'text': word['text'],
                        'start': word['start'] / 1000.0,  # Convert from ms to seconds
                        'end': word['end'] / 1000.0,     # Convert from ms to seconds
                        'type': 'word'
                    })
                
                # Define filler words
                filler_words = ["äh", "ähm"]
                
                # Create word-level SRT
                srt_file = os.path.join(file_dir, f"{file_name}.srt")
                create_word_level_srt(words, srt_file, remove_fillers=args.remove_fillers, 
                                     filler_words=filler_words, fps=args.fps, 
                                     fps_offset_start=args.fps_offset_start, 
                                     fps_offset_end=args.fps_offset_end)
                print(f"Word-level SRT saved to {file_name}.srt")
            else:
                print(f"Failed to get word-level data: {words_response.status_code} - {words_response.text}")
                # Try to get the entire transcript data instead
                transcript_url = f"https://api.assemblyai.com/v2/transcript/{transcript.id}"
                transcript_response = requests.get(transcript_url, headers=headers)
                
                if transcript_response.status_code == 200:
                    transcript_data = json.loads(transcript_response.text)
                    if 'words' in transcript_data and transcript_data['words']:
                        words = []
                        for word in transcript_data['words']:
                            words.append({
                                'text': word['text'],
                                'start': word['start'] / 1000.0,  # Convert from ms to seconds
                                'end': word['end'] / 1000.0,      # Convert from ms to seconds
                                'type': 'word'
                            })
                        
                        # Create word-level SRT
                        srt_file = os.path.join(file_dir, f"{file_name}.srt")
                        create_word_level_srt(words, srt_file, remove_fillers=args.remove_fillers, 
                                            filler_words=filler_words, fps=args.fps, 
                                            fps_offset_start=args.fps_offset_start, 
                                            fps_offset_end=args.fps_offset_end)
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