#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to transcribe audio with all APIs and analyze JSON outputs.
"""
import sys
import os
# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

from pathlib import Path
import json
from audio_transcribe.utils.api import get_api_instance
from audio_transcribe.utils.parsers import load_json_data, detect_and_parse_json

def test_api(api_name: str, audio_path: Path, force: bool = True):
    """Test a single API and return the result."""
    print(f"\n{'='*60}")
    print(f"Testing {api_name.upper()} API")
    print(f"{'='*60}")
    
    try:
        api_instance = get_api_instance(api_name)
        if not api_instance.check_api_key():
            print(f"[X] API key not found or invalid for {api_name}")
            return None
        
        print(f"[OK] API key valid for {api_name}")
        print(f"Transcribing {audio_path}...")
        
        result = api_instance.transcribe(audio_path, force=force)
        
        if result:
            print(f"[OK] Transcription successful")
            print(f"  - Words: {len(result.words)}")
            print(f"  - Text length: {len(result.text)}")
            return result
        else:
            print(f"[X] Transcription failed")
            return None
            
    except Exception as e:
        print(f"[X] Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_json(json_path: Path):
    """Analyze a JSON file to check if it has word-level data."""
    print(f"\nAnalyzing: {json_path.name}")
    
    try:
        data = load_json_data(json_path)
        if not data:
            print("  [X] Could not load JSON")
            return None
        
        analysis = {
            "file": json_path.name,
            "has_words": "words" in data and isinstance(data.get("words"), list),
            "word_count": len(data.get("words", [])) if "words" in data else 0,
            "has_text": "text" in data,
            "has_segments": "segments" in data,
            "api_name": data.get("api_name", "unknown"),
            "word_level_timestamps": False,
            "sample_word": None
        }
        
        if analysis["has_words"] and analysis["word_count"] > 0:
            first_word = data["words"][0]
            analysis["sample_word"] = first_word
            # Check if word has start/end timestamps
            has_start = "start" in first_word or "start_time" in first_word
            has_end = "end" in first_word or "end_time" in first_word
            analysis["word_level_timestamps"] = has_start and has_end
        
        return analysis
        
    except Exception as e:
        print(f"  [X] Error analyzing JSON: {e}")
        return None

def main():
    audio_path = Path("test/audio-test.mkv")
    
    if not audio_path.exists():
        print(f"[X] Audio file not found: {audio_path}")
        print("Will analyze existing JSON files only...")
    
    apis = ["assemblyai", "elevenlabs", "groq", "openai"]
    results = {}
    
    # Test all APIs (only if audio file exists and API keys are available)
    if audio_path.exists():
        for api_name in apis:
            result = test_api(api_name, audio_path, force=True)
            if result:
                results[api_name] = result
    else:
        print("\nSkipping API tests - analyzing existing JSON files only...")
    
    # Analyze JSON files
    print(f"\n{'='*60}")
    print("ANALYZING JSON FILES")
    print(f"{'='*60}")
    
    test_dir = Path("test")
    json_files = list(test_dir.glob(f"audio-test_*.json"))
    
    analyses = []
    for json_file in sorted(json_files):
        analysis = analyze_json(json_file)
        if analysis:
            analyses.append(analysis)
    
    # Print analysis results
    print(f"\n{'='*60}")
    print("ANALYSIS RESULTS")
    print(f"{'='*60}\n")
    
    for analysis in analyses:
        print(f"File: {analysis['file']}")
        print(f"   API: {analysis['api_name']}")
        print(f"   Has words array: {'[OK]' if analysis['has_words'] else '[X]'}")
        print(f"   Word count: {analysis['word_count']}")
        print(f"   Word-level timestamps: {'[OK]' if analysis['word_level_timestamps'] else '[X]'}")
        if analysis['sample_word']:
            print(f"   Sample word: {analysis['sample_word']}")
        print()
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY - Word-based SAT Export Support")
    print(f"{'='*60}\n")
    
    supported_apis = []
    for analysis in analyses:
        if analysis['word_level_timestamps'] and analysis['word_count'] > 0:
            supported_apis.append(analysis['api_name'])
            print(f"[OK] {analysis['api_name'].upper()}: Supports word-based export")
        else:
            print(f"[X] {analysis['api_name'].upper()}: Does NOT support word-based export")
    
    print(f"\n{'='*60}")
    print(f"Total APIs supporting word-based export: {len(supported_apis)}")
    print(f"APIs: {', '.join(supported_apis) if supported_apis else 'None'}")
    print(f"{'='*60}\n")
    
    # Check if documentation is correct
    print("DOCUMENTATION CHECK:")
    if "assemblyai" in supported_apis and len(supported_apis) == 1:
        print("[OK] Documentation is CORRECT: Only AssemblyAI supports word-based export")
    elif "assemblyai" in supported_apis and len(supported_apis) > 1:
        print("[!] Documentation is INCORRECT: Multiple APIs support word-based export")
        print(f"  Should update documentation to include: {', '.join(supported_apis)}")
    else:
        print("[!] Unexpected result: AssemblyAI not in supported list")

if __name__ == "__main__":
    main()

