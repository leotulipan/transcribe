#!/usr/bin/env python3
"""
Join two SRT files for DaVinci Resolve, with options to remove pause markers.

Usage:
    uv run join_srt_for_davinci.py file1.srt speaker1 file2.srt speaker2 [-o output.srt]
        [--remove-all-pauses]
        [--smart-pause-removal]
        [--pause-threshold-ms 500]

Options:
    --remove-all-pauses      Remove all lines where text is exactly (...)
    --smart-pause-removal    Remove pauses only when the next subtitle is from the other speaker and the gap is less than threshold
    --pause-threshold-ms     Threshold in ms for smart pause removal (default: 500)
    -o, --output             Output file (default: joined_for_davinci.srt)

Notes:
- Overlapping timecodes are allowed (SRT standard, DaVinci supports this).
- Pause marker is (...)
"""
import sys
import re
import argparse
from datetime import datetime, timedelta
from typing import List, Dict

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
    start_time, end_time = timecode_match.groups()
    try:
        start_dt = datetime.strptime(start_time, '%H:%M:%S,%f')
        end_dt = datetime.strptime(end_time, '%H:%M:%S,%f')
    except ValueError:
        return None
    text = '\n'.join(lines[2:]).strip()
    return {
        'index': index,
        'start_time': start_time,
        'end_time': end_time,
        'start_dt': start_dt,
        'end_dt': end_dt,
        'text': text,
        'speaker': speaker
    }

def parse_srt(file_path, speaker=None):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    blocks = re.split(r'\n\n+', content.strip())
    return [s for s in (parse_srt_block(b, speaker) for b in blocks) if s]

def format_time(dt: datetime) -> str:
    return dt.strftime('%H:%M:%S,%f')[:-3]

def write_srt(subs: List[Dict], output_file: str):
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, sub in enumerate(subs, 1):
            f.write(f"{idx}\n{sub['start_time']} --> {sub['end_time']}\n{sub['text']}\n\n")

def remove_all_pauses(subs: List[Dict]) -> List[Dict]:
    return [s for s in subs if s['text'].strip() != '(...)']

def remove_smart_pauses(subs: List[Dict], pause_threshold_ms: int) -> List[Dict]:
    result = []
    for i, sub in enumerate(subs):
        if sub['text'].strip() != '(...)':
            result.append(sub)
            continue
        # Only consider removing if next subtitle exists
        if i + 1 < len(subs):
            next_sub = subs[i + 1]
            # Remove pause if next is other speaker and gap is short
            if sub['speaker'] != next_sub['speaker']:
                gap = (next_sub['start_dt'] - sub['end_dt']).total_seconds() * 1000
                if 0 <= gap <= pause_threshold_ms:
                    continue  # Remove this pause
        result.append(sub)
    return result

def main():
    parser = argparse.ArgumentParser(description='Join two SRT files for DaVinci with pause marker options.')
    parser.add_argument('file1', help='First SRT file')
    parser.add_argument('speaker1', help='Speaker name for first file')
    parser.add_argument('file2', help='Second SRT file')
    parser.add_argument('speaker2', help='Speaker name for second file')
    parser.add_argument('-o', '--output', default='joined_for_davinci.srt', help='Output file')
    parser.add_argument('--remove-all-pauses', action='store_true', help='Remove all pauses')
    parser.add_argument('--smart-pause-removal', action='store_true', help='Remove only pauses when other speaker starts after short gap')
    parser.add_argument('--pause-threshold-ms', type=int, default=500, help='Threshold in ms for smart pause removal (default: 500)')
    args = parser.parse_args()

    subs1 = parse_srt(args.file1, args.speaker1)
    subs2 = parse_srt(args.file2, args.speaker2)
    all_subs = subs1 + subs2
    all_subs.sort(key=lambda s: s['start_dt'])

    if args.remove_all_pauses:
        all_subs = remove_all_pauses(all_subs)
    elif args.smart_pause_removal:
        all_subs = remove_smart_pauses(all_subs, args.pause_threshold_ms)

    write_srt(all_subs, args.output)
    print(f"Joined SRT written to {args.output}")

if __name__ == '__main__':
    main() 