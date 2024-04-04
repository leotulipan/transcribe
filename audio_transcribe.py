import re
from dotenv import load_dotenv
import sys
import os
import pprint
import calendar
from datetime import datetime
import json
import requests
from datetime import date, timedelta
import json
import argparse

from openai import OpenAI
import pydub
from pydub import AudioSegment
# Docs https://github.com/jiaaro/pydub/blob/master/API.markdown

# global variables
args = ""


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
    return os.path.exists(transcript_path)

def main():
    global args
    parser = argparse.ArgumentParser()
    # debug 
    parser.add_argument("-d", "--debug", help="Debug mode", action="store_true")

    parser.add_argument("-f", "--file", required=False, help="filename, foldername or pattern to transcribe")

    # k / keep parameter to keep the ogg files
    parser.add_argument("-k", "--keep", help="Keep the ogg files", action="store_true")
    args = parser.parse_args()

    pp = pprint.PrettyPrinter(indent=4)

    client = OpenAI()

    if in_debug_mode():
        clear_screen()
        print("We are in debug mode.") 

    load_dotenv() # load environment variables from .env file
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Initialize an empty dictionary to store the files
    files_dict = {}

    # Check if args.file is a directory
    if os.path.isdir(args.file):
        # If it's a directory, get all audio and video files in the directory
        for root, dirs, files in os.walk(args.file):
            for file in files:
                if file.endswith(('.mp3', '.wav', '.ogg', '.mp4', '.avi', '.mp4', '.mov', '.mkv', '.webm', '.m4a', '.flac', '.aac', '.wma', '.aiff', '.flv', '.wmv', '.3gp', '.3g2', '.m4v', '.ts', '.m2ts', '.mts', '.vob', '.ogv', '.ogg', '.oga', '.opus', '.spx', '.amr', '.mka', '.mk3d')):
                    files_dict[file] = os.path.join(root, file)
    # Check if args.file is a file
    elif os.path.isfile(args.file):
        # If it's a file, just add it to the dictionary
        files_dict[args.file] = args.file
    # Check if args.file is a wildcard pattern
    elif '*' in args.file or '?' in args.file:
        # If it's a wildcard pattern, find all matching files
        for file in glob.glob(args.file):
            files_dict[file] = file
    else:
        print("Invalid input. Please provide a valid file, directory, or wildcard pattern.")
        return
    
    for file_name, file_path in files_dict.items():
        # check if file exists
        if not os.path.exists(file_path):
            print(f"Audio File {file_name} does not exist!")
            continue

        full_file_name = os.path.basename(file_path)
        file_name, file_extension = os.path.splitext(full_file_name)    

        # cd to file_path
        os.chdir(os.path.dirname(file_path))

        # Check if transcript exists
        if check_transcript_exists(os.path.dirname(file_path), file_name):
            print(f"Transcript for {file_name} exists!")
            continue

  
        # https://platform.openai.com/docs/guides/speech-to-text/longer-inputs
        # G:\Geteilte Ablagen\_3_References\Transscribe Queue\v12044gd0000cg3j4tbc77ubn6e7us3g.mp4
        audio = AudioSegment.from_file(file_path)

        # print("frame_rate: ", audio.frame_rate)
        # print("frame_count: ", audio.frame_count()  )
        #print(f"Audio length: {audio.duration_seconds}")
        # LengthInSec = audio.frame_count() /  audio.frame_rate
        # print(f"Audio length: {LengthInSec}")

        # raw_date byte length
        if in_debug_mode():
            mb = len(audio.raw_data) / (1024.0 * 1024.0)
            print(f"Audio size: {mb} MB")
            
        # https://platform.openai.com/docs/guides/speech-to-text/longer-inputs
        # PyDub handles time in milliseconds
        twenty_four_minutes = 24 * 60 * 1000

        # if duration is less than twenty_four_minutes, then we don't need to split
        if audio.duration_seconds < twenty_four_minutes and mb < 15:
            chunks = [audio]
        else:
            # array with chunks for every twenty_four_minutes
            chunks = [audio[i:i+twenty_four_minutes] for i in range(0, len(audio), twenty_four_minutes)]
        if in_debug_mode():
            print(f"Number of chunks: {len(chunks)}")

        #mono = audio.set_channels(1)

        transcript = ""

        # loop through chunks and export each one as a .ogg file
        for i, chunk in enumerate(chunks):

            if mb < 15:
                audio_file = open(file_path, "rb")
            else:
                # export as .ogg
                compressed_filename = f"{file_name}_{i}.ogg"
                # skip if file exists
                if not os.path.exists(compressed_filename):
                    if in_debug_mode():
                        print(f"Compressing {compressed_filename}")            
                    chunk.export(compressed_filename, codec="libvorbis",
                                        format="ogg",
                                        bitrate="128k",
                                        parameters=["-ac", "1", "-q", "0"])
                elif in_debug_mode():
                        print(f"File {compressed_filename} exists. Skipping compression.")
                print(f"Transcribing {file_name} - chunk {i}")
                # add to transscript text
                audio_file =  open(compressed_filename, "rb")

            transcription = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file, 
                response_format="text"
            )
            #print(transcription)

            transcript += transcription
            # old 0.28 api / transcript += openai.Audio.transcribe("whisper-1", transcript_file, response_format="text")
            #close transscript file
            audio_file.close()
            # remove temp ogg file
            if not args.keep and os.path.exists(compressed_filename):
                    os.remove(compressed_filename)
            
        # Save the transcript
        file_dir = os.path.dirname(file_path)
        with open(os.path.join(file_dir, f"{file_name}.txt"), "w") as f:
            f.write(transcript)
        print(f"Transcript: {transcript[:160]} saved to {file_name}.txt")
        
if __name__ == '__main__':
    main()        