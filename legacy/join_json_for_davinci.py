#!/usr/bin/env python3
"""
Join two ElevenLabs JSON files with word lists for DaVinci Resolve.

Usage:
    uv run join_json_for_davinci.py --file1 file1.json --file2 file2.json [--speaker1 "Alice"] [--speaker2 "Bob"] [-o output.srt] [--chars-per-line 45]

Options:
    --file1                  First JSON file (required)
    --file2                  Second JSON file (required)
    --speaker1               Speaker name for first file (default: "Speaker 1")
    --speaker2               Speaker name for second file (default: "Speaker 2")
    --chars-per-line         Maximum characters per line (default: 45)
    -o, --output             Output file (default: joined_for_davinci.srt)

Notes:
- Processes ElevenLabs JSON format with "words" array
- Ignores spacing elements and pauses for SRT output
- Speaker names are added to the front of each subtitle
- Groups words into lines based on chars-per-line limit
"""
import sys
import json
import argparse
import os
from datetime import datetime
from typing import List, Dict, NamedTuple

class Word(NamedTuple):
    speaker: str
    text: str
    start_time: float
    end_time: float
    is_spacing: bool

def format_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

def parse_json_words(file_path: str, speaker: str) -> List[Word]:
    """Parse ElevenLabs JSON file and extract words."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    words = []
    if 'words' in data:
        for word_data in data['words']:
            is_spacing = word_data.get('type') == 'spacing'
            words.append(Word(
                speaker=speaker,
                text=word_data['text'],
                start_time=word_data['start'],
                end_time=word_data['end'],
                is_spacing=is_spacing
            ))
    
    return words

def group_words_into_lines(words: List[Word], chars_per_line: int) -> List[Dict]:
    """Group words into subtitle lines based on character limit."""
    if not words:
        return []
    
    # Filter out spacing elements
    speech_words = [w for w in words if not w.is_spacing]
    if not speech_words:
        return []
    
    lines = []
    current_line_words = []
    current_line_text = ""
    current_start = None
    
    for word in speech_words:
        word_text = word.text.strip()
        if not word_text:
            continue
            
        # Check if adding this word would exceed the character limit
        test_text = current_line_text + (" " if current_line_text else "") + word_text
        
        if len(test_text) > chars_per_line and current_line_words:
            # Save current line
            lines.append({
                'speaker': current_line_words[0].speaker,
                'text': current_line_text.strip(),
                'start_time': current_start,
                'end_time': current_line_words[-1].end_time
            })
            
            # Start new line
            current_line_words = [word]
            current_line_text = word_text
            current_start = word.start_time
        else:
            # Add word to current line
            if not current_line_words:
                current_start = word.start_time
            current_line_words.append(word)
            current_line_text = test_text
    
    # Add final line if exists
    if current_line_words:
        lines.append({
            'speaker': current_line_words[0].speaker,
            'text': current_line_text.strip(),
            'start_time': current_start,
            'end_time': current_line_words[-1].end_time
        })
    
    return lines

def write_srt(lines: List[Dict], output_file: str):
    """Write subtitle lines to SRT file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, line in enumerate(lines, 1):
            start_str = format_time(line['start_time'])
            end_str = format_time(line['end_time'])
            text = f"{line['speaker']}: {line['text']}"
            f.write(f"{idx}\n{start_str} --> {end_str}\n{text}\n\n")

def main():
    parser = argparse.ArgumentParser(description='Join two ElevenLabs JSON files for DaVinci.')
    parser.add_argument('--file1', required=True, help='First JSON file')
    parser.add_argument('--file2', required=True, help='Second JSON file')
    parser.add_argument('--speaker1', default='Speaker 1', help='Speaker name for first file (default: "Speaker 1")')
    parser.add_argument('--speaker2', default='Speaker 2', help='Speaker name for second file (default: "Speaker 2")')
    parser.add_argument('--chars-per-line', type=int, default=45, help='Maximum characters per line (default: 45)')
    parser.add_argument('-o', '--output', default='joined_for_davinci.srt', help='Output file')
    args = parser.parse_args()

    # Determine output path
    if args.output == 'joined_for_davinci.srt':
        out_dir = os.path.dirname(os.path.abspath(args.file1))
        output_file = os.path.join(out_dir, 'joined_for_davinci.srt')
    else:
        # If output is just a filename (no path), save in file1's dir
        if os.path.dirname(args.output):
            output_file = args.output
        else:
            out_dir = os.path.dirname(os.path.abspath(args.file1))
            output_file = os.path.join(out_dir, args.output)

    # Parse both JSON files
    words1 = parse_json_words(args.file1, args.speaker1)
    words2 = parse_json_words(args.file2, args.speaker2)
    
    # Combine and sort by start time
    all_words = words1 + words2
    all_words.sort(key=lambda w: w.start_time)
    
    # Group words into subtitle lines
    lines = group_words_into_lines(all_words, args.chars_per_line)
    
    # Write SRT file
    write_srt(lines, output_file)
    print(f"Joined SRT written to {output_file}")
    print(f"Total subtitle lines: {len(lines)}")

if __name__ == '__main__':
    main() 