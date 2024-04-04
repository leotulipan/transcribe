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

import replicate
import base64

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
    parser.add_argument("-f", "--file", required=True, help="filename to transcribe")
    args = parser.parse_args()

    pp = pprint.PrettyPrinter(indent=4)

    client = OpenAI()

    if in_debug_mode():
        clear_screen()
        print("We are in debug mode.") 

    load_dotenv() # load environment variables from .env file
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # check if file exists
    if not os.path.exists(args.file):
        print("Audio File does not exist!")
        return

    file_path, full_file_name = os.path.split(args.file)
    file_name, file_extension = os.path.splitext(full_file_name)    

    # cd to file_path
    os.chdir(file_path)

    # Check if transcript exists
    if check_transcript_exists(file_path, file_name):
        print("Transcript exists!")
        return    


    # https://platform.openai.com/docs/guides/speech-to-text/longer-inputs
    # G:\Geteilte Ablagen\_3_References\Transscribe Queue\v12044gd0000cg3j4tbc77ubn6e7us3g.mp4
    input_audio = AudioSegment.from_file(args.file)

    # raw_date byte length
    if in_debug_mode():
        mb = len(input_audio.raw_data) / (1024.0 * 1024.0)
        print(f"Audio size: {mb} MB")
        
    audio_base64 = base64.b64encode(open(args.file, "rb").read()).decode('utf-8')

    # docs https://replicate.com/thomasmol/whisper-diarization
    transcript = replicate.run(
        "thomasmol/whisper-diarization:b9fd8313c0d492bf1ce501b3d188f945389327730773ec1deb6ef233df6ea119",
        input = {
            "file": open(args.file, "rb"),
            "prompt": "SPEAKER_01: Leonard and SPEAKER_02: Julia.",
            #"num_speakers": 2,
            "group_segments": True,
            # Language of the spoken words as a language code like 'en'. Leave empty to auto detect language.
            #"language": "de",
            "offset_seconds": 0,
            "transcript_output_format": "segments_only"
        }
    )
    
        # transcription = client.audio.transcriptions.create(
        #     model="whisper-1", 
        #     file=audio_file, 
        #     response_format="text"
        # )
        # #print(transcription)

        # transcript += transcription
        # # old 0.28 api / transcript += openai.Audio.transcribe("whisper-1", transcript_file, response_format="text")
        # #close transscript file
        # audio_file.close()
        
    if in_debug_mode():
        print(f"Number of segments: {len(transcript['segments'])}")
        # print(f"Predict time: {transcript['metrics']['predict_time']}")
        # print(f"Cost: {transcript['metrics']['predict_time'] * 0.000575}")

    # Format the transcript as .srt subtitles
    srt_format = ""
    for i, segment in enumerate(transcript['segments']):
        start_time = str(timedelta(seconds=float(segment['start'])))
        end_time = str(timedelta(seconds=float(segment['end'])))
        srt_format += f"{i+1}\n{start_time} --> {end_time}\n{segment['speaker']}: {segment['text']}\n\n"

    # Save the .srt subtitles
    with open(os.path.join(file_path, f"{file_name}.txt"), "w") as f:
        f.write(srt_format)
    #print(f"Transcript first segment: {transcript.output.segments[0]}")

    # # Save the transcript
    # with open(os.path.join(file_path, f"{file_name}.txt"), "w") as f:
    #     f.write(json.dump(transcript, f, indent=4))
        
if __name__ == '__main__':
    main()        