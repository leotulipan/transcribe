#!/usr/bin/env python3
"""
Simple script to convert AssemblyAI JSON to SRT
Usage: python json_to_srt.py <json_file>
"""
import sys
import json
from pathlib import Path

# Add the parent directory to the path so we can import from audio_transcribe
sys.path.insert(0, str(Path(__file__).parent))

from audio_transcribe.utils.parsers import parse_assemblyai_format, load_json_data
from audio_transcribe.utils.formatters import create_srt_file

def main():
    if len(sys.argv) < 2:
        print("Usage: python json_to_srt.py <json_file>")
        sys.exit(1)
    
    json_path = Path(sys.argv[1])
    
    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)
    
    print(f"Loading JSON from: {json_path}")
    json_data = load_json_data(json_path)
    
    print("Parsing AssemblyAI format...")
    result = parse_assemblyai_format(json_data)
    
    print(f"Found {len(result.words)} words")
    
    # Create SRT file with same name as JSON
    srt_path = json_path.with_suffix('.srt')
    
    print(f"Creating SRT file: {srt_path}")
    create_srt_file(result, srt_path, format_type="standard", start_hour=0)
    
    print(f"✓ Successfully created {srt_path}")

if __name__ == "__main__":
    main()
