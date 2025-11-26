# -*- coding: utf-8 -*-
"""
Analyze existing JSON files to check word-level timestamp support.
"""
import sys
import json
from pathlib import Path

def analyze_json(json_path: Path):
    """Analyze a JSON file to check if it has word-level data."""
    print(f"\nAnalyzing: {json_path.name}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        analysis = {
            "file": json_path.name,
            "has_words": "words" in data and isinstance(data.get("words"), list),
            "word_count": len(data.get("words", [])) if "words" in data else 0,
            "has_text": "text" in data,
            "has_segments": "segments" in data,
            "api_name": data.get("api_name", "unknown"),
            "word_level_timestamps": False,
            "sample_word": None,
            "word_structure": None
        }
        
        # Try to detect API from filename
        if analysis["api_name"] == "unknown":
            filename_lower = json_path.name.lower()
            if "assemblyai" in filename_lower:
                analysis["api_name"] = "assemblyai"
            elif "elevenlabs" in filename_lower:
                analysis["api_name"] = "elevenlabs"
            elif "groq" in filename_lower:
                analysis["api_name"] = "groq"
            elif "openai" in filename_lower:
                analysis["api_name"] = "openai"
        
        if analysis["has_words"] and analysis["word_count"] > 0:
            first_word = data["words"][0]
            analysis["sample_word"] = first_word
            analysis["word_structure"] = list(first_word.keys())
            # Check if word has start/end timestamps
            has_start = "start" in first_word or "start_time" in first_word
            has_end = "end" in first_word or "end_time" in first_word
            analysis["word_level_timestamps"] = has_start and has_end
        
        return analysis
        
    except Exception as e:
        print(f"  [X] Error analyzing JSON: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    test_dir = Path("test")
    
    if not test_dir.exists():
        print(f"[X] Test directory not found: {test_dir}")
        return
    
    # Find all JSON files
    json_files = list(test_dir.glob(f"audio-test*.json"))
    
    if not json_files:
        print(f"[X] No JSON files found in {test_dir}")
        return
    
    print(f"{'='*60}")
    print("ANALYZING JSON FILES")
    print(f"{'='*60}")
    print(f"Found {len(json_files)} JSON files\n")
    
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
        if analysis['word_structure']:
            print(f"   Word structure keys: {', '.join(analysis['word_structure'])}")
        if analysis['sample_word']:
            # Show a simplified version of sample word
            sample = {k: v for k, v in analysis['sample_word'].items() if k in ['text', 'start', 'end', 'type', 'word']}
            print(f"   Sample word: {sample}")
        print()
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY - Word-based SAT/SRT Export Support")
    print(f"{'='*60}\n")
    
    supported_apis = []
    for analysis in analyses:
        if analysis['word_level_timestamps'] and analysis['word_count'] > 0:
            if analysis['api_name'] not in supported_apis:
                supported_apis.append(analysis['api_name'])
            print(f"[OK] {analysis['api_name'].upper()}: Supports word-based export")
            print(f"     File: {analysis['file']}")
        else:
            print(f"[X] {analysis['api_name'].upper()}: Does NOT support word-based export")
            print(f"     File: {analysis['file']}")
        print()
    
    print(f"{'='*60}")
    print(f"Total APIs supporting word-based export: {len(supported_apis)}")
    print(f"APIs: {', '.join(supported_apis) if supported_apis else 'None'}")
    print(f"{'='*60}\n")
    
    # Check if documentation is correct
    print("DOCUMENTATION CHECK:")
    print(f"{'='*60}")
    if "assemblyai" in supported_apis and len(supported_apis) == 1:
        print("[OK] Documentation is CORRECT: Only AssemblyAI supports word-based export")
    elif "assemblyai" in supported_apis and len(supported_apis) > 1:
        print("[!] Documentation is INCORRECT: Multiple APIs support word-based export")
        print(f"  Should update documentation to include: {', '.join(supported_apis)}")
        print(f"  Currently documentation says only AssemblyAI supports it.")
    elif "assemblyai" not in supported_apis:
        print("[!] Unexpected result: AssemblyAI not in supported list")
        print(f"  Supported APIs: {', '.join(supported_apis) if supported_apis else 'None'}")
    else:
        print(f"[OK] Documentation might need update. Supported: {', '.join(supported_apis)}")

if __name__ == "__main__":
    main()

