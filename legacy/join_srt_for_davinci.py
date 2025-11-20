#!/usr/bin/env python3
"""
Join two SRT files for DaVinci Resolve, with options to remove pause markers.

Usage:
    uv run join_srt_for_davinci.py --file1 file1.srt --file2 file2.srt [--speaker1 "Alice"] [--speaker2 "Bob"] [-o output.srt] [--remove-all-pauses]

Options:
    --file1                  First SRT file (required)
    --file2                  Second SRT file (required)
    --speaker1               Speaker name for first file (default: "Speaker 1")
    --speaker2               Speaker name for second file (default: "Speaker 2")
    --remove-all-pauses      Remove all lines where text is exactly (...)
    -o, --output             Output file (default: joined_for_davinci.srt)

Notes:
- Overlapping timecodes are allowed (SRT standard, DaVinci supports this).
- Pause marker is (...)
- Algorithm automatically removes overlapping pauses when other speaker is talking
- Speaker names are added to the front of each subtitle (except pauses)
"""
import sys
import re
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, NamedTuple
import os

class Subtitle(NamedTuple):
    speaker: str
    text: str
    is_pause: bool
    start_time: datetime
    end_time: datetime
    start_time_str: str
    end_time_str: str

def parse_srt_block(block, speaker=None):
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
    start_time_str, end_time_str = timecode_match.groups()
    try:
        start_dt = datetime.strptime(start_time_str, '%H:%M:%S,%f')
        end_dt = datetime.strptime(end_time_str, '%H:%M:%S,%f')
    except ValueError:
        return None
    text = '\n'.join(lines[2:]).strip()
    is_pause = text == '(...)'
    
    return Subtitle(
        speaker=speaker,
        text=text,
        is_pause=is_pause,
        start_time=start_dt,
        end_time=end_dt,
        start_time_str=start_time_str,
        end_time_str=end_time_str
    )

def parse_srt(file_path, speaker=None):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    blocks = re.split(r'\n\n+', content.strip())
    return [s for s in (parse_srt_block(b, speaker) for b in blocks) if s]

def write_srt(subs: List[Subtitle], output_file: str):
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, sub in enumerate(subs, 1):
            # Add speaker name to front of text (except for pauses)
            if sub.is_pause:
                text = sub.text
            else:
                text = f"{sub.speaker}: {sub.text}"
            f.write(f"{idx}\n{sub.start_time_str} --> {sub.end_time_str}\n{text}\n\n")

def remove_all_pauses(subs: List[Subtitle]) -> List[Subtitle]:
    """Remove all pause markers."""
    return [s for s in subs if not s.is_pause]

def join_consecutive_pauses(subs: List[Subtitle]) -> List[Subtitle]:
    """Join consecutive pauses from the same speaker when no other subtitle is between them."""
    if not subs:
        return subs
    
    result = []
    i = 0
    while i < len(subs):
        current = subs[i]
        if not current.is_pause:
            result.append(current)
            i += 1
            continue
        
        # Found a pause, look for consecutive pauses from same speaker
        consecutive_pauses = [current]
        j = i + 1
        while j < len(subs) and subs[j].is_pause and subs[j].speaker == current.speaker:
            consecutive_pauses.append(subs[j])
            j += 1
        
        if len(consecutive_pauses) > 1:
            # Merge consecutive pauses
            merged_pause = Subtitle(
                speaker=current.speaker,
                text='(...)',
                is_pause=True,
                start_time=consecutive_pauses[0].start_time,
                end_time=consecutive_pauses[-1].end_time,
                start_time_str=consecutive_pauses[0].start_time_str,
                end_time_str=consecutive_pauses[-1].end_time_str
            )
            result.append(merged_pause)
        else:
            result.append(current)
        
        i = j
    
    return result

def remove_overlapping_pauses(subs: List[Subtitle]) -> List[Subtitle]:
    """Remove pauses that overlap with speech from other speaker."""
    if not subs:
        return subs
    
    result = []
    for i, current in enumerate(subs):
        if not current.is_pause:
            result.append(current)
            continue
        
        # Check if this pause overlaps with speech from other speaker
        should_remove = False
        for other in subs:
            if (other.speaker != current.speaker and 
                not other.is_pause and 
                current.start_time < other.end_time and 
                current.end_time > other.start_time):
                # Pause overlaps with other speaker's speech
                should_remove = True
                break
        
        if not should_remove:
            result.append(current)
    
    return result

def validate_and_clean(subs: List[Subtitle]) -> List[Subtitle]:
    """Additional robustness: remove empty subtitles, sort by time, handle edge cases."""
    # Remove empty or invalid subtitles
    valid_subs = [s for s in subs if s.text.strip() and s.start_time < s.end_time]
    
    # Sort by start time
    valid_subs.sort(key=lambda x: x.start_time)
    
    return valid_subs

def main():
    parser = argparse.ArgumentParser(description='Join two SRT files for DaVinci with pause marker options.')
    parser.add_argument('--file1', required=True, help='First SRT file')
    parser.add_argument('--file2', required=True, help='Second SRT file')
    parser.add_argument('--speaker1', default='Speaker 1', help='Speaker name for first file (default: "Speaker 1")')
    parser.add_argument('--speaker2', default='Speaker 2', help='Speaker name for second file (default: "Speaker 2")')
    parser.add_argument('-o', '--output', default='joined_for_davinci.srt', help='Output file')
    parser.add_argument('--remove-all-pauses', action='store_true', help='Remove all pauses')
    args = parser.parse_args()

    # Determine output path
    if args.output is None:
        out_dir = os.path.dirname(os.path.abspath(args.file1))
        output_file = os.path.join(out_dir, 'joined_for_davinci.srt')
    else:
        # If output is just a filename (no path), save in file1's dir
        if os.path.dirname(args.output):
            output_file = args.output
        else:
            out_dir = os.path.dirname(os.path.abspath(args.file1))
            output_file = os.path.join(out_dir, args.output)

    # Parse both SRT files and combine into unified data structure
    subs1 = parse_srt(args.file1, args.speaker1)
    subs2 = parse_srt(args.file2, args.speaker2)
    all_subs = subs1 + subs2
    
    # Step 1: Validate and clean data
    all_subs = validate_and_clean(all_subs)
    
    # Step 2: Remove all pauses if requested
    if args.remove_all_pauses:
        all_subs = remove_all_pauses(all_subs)
    else:
        # Step 3: Join consecutive pauses
        all_subs = join_consecutive_pauses(all_subs)
        
        # Step 4: Remove overlapping pauses when other speaker is talking
        all_subs = remove_overlapping_pauses(all_subs)
    
    # Final sort by time (in case processing changed order)
    all_subs.sort(key=lambda s: s.start_time)
    
    write_srt(all_subs, output_file)
    print(f"Joined SRT written to {output_file}")
    print(f"Total subtitles: {len(all_subs)} (pauses: {sum(1 for s in all_subs if s.is_pause)})")

if __name__ == '__main__':
    main() 