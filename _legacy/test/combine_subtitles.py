#!/usr/bin/env python3
"""
Combines two SRT files, adds speaker names, sorts by start time, and outputs a combined SRT file.

Usage:
    python script_name.py <file1> <speaker1> <file2> <speaker2> [-o <output_file>] [--no-fix-timings]

Arguments:
    file1       : Path to the first SRT file.
    speaker1    : Speaker name for the first file.
    file2       : Path to the second SRT file.
    speaker2    : Speaker name for the second file.
    -o, --output: Output file name (default: combined.srt).
    --no-fix-timings: Skip automatic timing fixes.

Example:
    python combine_srts.py sub1.srt "Alice" sub2.srt "Bob" -o combined.srt

"""
# /// script
# dependencies = [
#   "pandas",
# ]
# ///

# buggy: it is not weaving correctly. all times start at 0
# https://claude.ai/chat/71d38e50-593e-4029-b48a-84704c7b6cf6
# https://gemini.google.com/app/b23503853587868d

import sys
import re
import argparse
from datetime import datetime, timedelta
import pandas as pd

def parse_srt_block(block, speaker=None):
    """Parse a single SRT block and return a dictionary."""
    lines = block.strip().split('\n')
    if len(lines) < 3:
        return None

    try:
        index = int(lines[0])
    except ValueError:
        return None

    timecode_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})', lines[1])
    if not timecode_match:
        return None

    start_time, end_time = timecode_match.groups()

    try:
        start_dt = datetime.strptime(start_time, '%H:%M:%S,%f')
        end_dt = datetime.strptime(end_time, '%H:%M:%S,%f')
    except ValueError:
        return None

    text = '\n'.join(lines[2:])
    if speaker:
        text = f"{speaker}: {text}"

    return {
        'index': index,
        'start_time': start_time,
        'end_time': end_time,
        'start_dt': start_dt,
        'end_dt': end_dt,
        'text': text
    }


def parse_srt(file_path, speaker=None):
    """Parse an SRT file and return a list of subtitle dictionaries."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = re.split(r'\n\n+', content.strip())
    subtitles = []
    for block in blocks:
        subtitle = parse_srt_block(block, speaker)
        if subtitle:
            subtitles.append(subtitle)
    return subtitles


def fix_timings(subtitles):
    """Fix potentially invalid timestamps (e.g., all starting at 00:00:00).
       Only fix if more than 30% of start times are identical.  Otherwise,
       assume the timings are intentional, even if unusual.
    """
    if not subtitles:
        return []

    # Check if a significant portion of subtitles have the same start time
    start_times = [sub['start_dt'] for sub in subtitles]
    most_common_time = max(set(start_times), key=start_times.count)
    if start_times.count(most_common_time) > len(subtitles) * 0.3:
        print("Warning: Many subtitles have the same start time. Attempting to fix timings...")

        # Sort by original index, assuming the file order was correct
        subtitles.sort(key=lambda x: x['index'])

        # Calculate duration of each subtitle, use it to adjust the next one.
        for i in range(len(subtitles)):
          if i == 0:  # first one starts at 0
              new_start_seconds = 0
          else:
              prev_sub = subtitles[i-1]
              new_start_seconds = (prev_sub['end_dt'] - datetime.min).total_seconds()

          duration = (subtitles[i]['end_dt'] - subtitles[i]['start_dt']).total_seconds()
          new_end_seconds = new_start_seconds + duration
          subtitles[i]['start_time'] = str(timedelta(seconds=new_start_seconds))
          if '.' not in subtitles[i]['start_time']:
            subtitles[i]['start_time'] += '.000'
          
          subtitles[i]['start_time'] = subtitles[i]['start_time'].replace('.', ',')[:12]


          subtitles[i]['end_time'] = str(timedelta(seconds=new_end_seconds))
          if '.' not in subtitles[i]['end_time']:
              subtitles[i]['end_time'] += '.000'
          subtitles[i]['end_time'] = subtitles[i]['end_time'].replace('.', ',')[:12]

          # Crucial: Update start_dt and end_dt AFTER string conversion
          subtitles[i]['start_dt'] = datetime.strptime(subtitles[i]['start_time'], '%H:%M:%S,%f')
          subtitles[i]['end_dt'] = datetime.strptime(subtitles[i]['end_time'], '%H:%M:%S,%f')

    return subtitles


def subtitles_to_dataframe(subtitles):
    """Convert a list of subtitle dictionaries to a Pandas DataFrame."""
    return pd.DataFrame(subtitles)


def dataframe_to_srt(df, output_file):
    """Export a DataFrame to an SRT file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for index, row in df.iterrows():
            f.write(f"{index + 1}\n")
            f.write(f"{row['start_time']} --> {row['end_time']}\n")
            f.write(f"{row['text']}\n\n")


def combine_subtitles(file1, speaker1, file2, speaker2, output_file, fix_timing=True):
    """Combine two SRT files, add speaker names, sort, and export to SRT."""
    subs1 = parse_srt(file1, speaker1)
    subs2 = parse_srt(file2, speaker2)

    if fix_timing:
        subs1 = fix_timings(subs1)
        subs2 = fix_timings(subs2)


    # Combine DataFrames *before* fixing timings if fix_timing is false
    df1 = subtitles_to_dataframe(subs1)
    df2 = subtitles_to_dataframe(subs2)
    combined_df = pd.concat([df1, df2])

    # Sort and reset index
    combined_df = combined_df.sort_values(by='start_dt').reset_index(drop=True)
    dataframe_to_srt(combined_df, output_file)

def main():
    parser = argparse.ArgumentParser(description='Combine two SRT files with speaker names.')
    parser.add_argument('file1', help='First SRT file')
    parser.add_argument('speaker1', help='Speaker name for the first file')
    parser.add_argument('file2', help='Second SRT file')
    parser.add_argument('speaker2', help='Speaker name for the second file')
    parser.add_argument('-o', '--output', default='combined.srt', help='Output file (default: combined.srt)')
    parser.add_argument('--no-fix-timings', action='store_true', help='Skip automatic timing fixes')

    args = parser.parse_args()

    try:
        combine_subtitles(args.file1, args.speaker1, args.file2, args.speaker2, args.output, not args.no_fix_timings)
        print(f"Successfully combined subtitles into {args.output}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()