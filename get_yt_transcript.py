#!/usr/bin/env python3
# /// script
# dependencies = [
#   "yt-dlp",
# ]
# ///
import yt_dlp
import os
import sys

def download_subtitles(url, output_file="transcript.txt"):
    """Downloads subtitles from a YouTube video using yt-dlp.

    Args:
        url: The YouTube video URL.
        output_file: The name of the file to save the transcript to.
    """

    ydl_opts = {
        'skip_download': True,
        'writesubtitles': True,
        'subtitleslangs': ['original'],
        'subtitlesformat': 'srt',
        'outtmpl': output_file,
        'ignoreerrors': True,
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            # Check if 'subtitles' key exists and is not None
            if 'subtitles' in info and info['subtitles'] is not None and any(info['subtitles'].values()):
                if ydl.download([url]) == 0 and os.path.exists(output_file):
                    print(f"Transcript saved to {output_file}")
                    return  # Exit the function if successful

        print("No original subtitles found. Trying auto-generated English...")
        ydl_opts['writesubtitles'] = False
        ydl_opts['writeautomaticsub'] = True
        ydl_opts['subtitleslangs'] = ['en']

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False) # Get info again for auto-subs
            # Check if 'automatic_captions' key exists
            if 'automatic_captions' in info and info['automatic_captions'] is not None and any(info['automatic_captions'].values()):
                if ydl.download([url]) == 0 and os.path.exists(output_file):
                    print(f"Transcript saved to {output_file}")
                    return

        print("No auto-generated English subtitles found. Trying en-orig...")
        ydl_opts['writeautomaticsub'] = False
        ydl_opts['writesubtitles'] = True
        ydl_opts['subtitleslangs'] = ['en-orig']

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)  # Get info again for en-orig
             # Check if 'subtitles' key exists and is not None
            if 'subtitles' in info and info['subtitles'] is not None and any(info['subtitles'].values()):
                if ydl.download([url]) == 0 and os.path.exists(output_file):
                    print(f"Transcript saved to {output_file}")
                    return


        print("No en-orig found. Listing all available subtitles...\n")
        ydl_opts_list = {
            'skip_download': True,
            'listsubtitles': True,
            'quiet': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts_list) as ydl_list:
            ydl_list.download([url])
        sys.exit(1)  # Exit after listing subtitles


    except yt_dlp.utils.DownloadError as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


def main():
    """Gets the YouTube URL from the user or command line."""
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = input("Enter YouTube URL: ")

    if not url:
        print("No URL provided. Exiting.")
        sys.exit(1)

    download_subtitles(url)

if __name__ == "__main__":
    main()