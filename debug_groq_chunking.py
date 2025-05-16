import os
import json
import argparse
import subprocess
from datetime import datetime

def save_temp_json(data, output_path):
    """Save JSON data to file for debugging"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved debug info to: {output_path}")

def analyze_chunks(chunk_files):
    """Analyze the timestamp gaps between chunks"""
    all_chunks = []
    
    # Load all chunk data
    for file in chunk_files:
        with open(file, 'r', encoding='utf-8') as f:
            chunk_data = json.load(f)
            # Extract information about the chunk
            chunk_info = {
                'file': file,
                'words_count': len(chunk_data.get('words', [])),
                'first_word': chunk_data.get('words', [{}])[0] if chunk_data.get('words') else None,
                'last_word': chunk_data.get('words', [{}])[-1] if chunk_data.get('words') else None,
                'start_time': float(chunk_data.get('start_time', 0)),
                'end_time': float(chunk_data.get('end_time', 0)),
            }
            all_chunks.append(chunk_info)
    
    # Sort chunks by start time
    all_chunks.sort(key=lambda x: x['start_time'])
    
    # Analyze gaps between chunks
    print("\nChunk Analysis:")
    print("=" * 80)
    
    for i, chunk in enumerate(all_chunks):
        print(f"Chunk {i+1}: {chunk['file']}")
        print(f"  Words count: {chunk['words_count']}")
        print(f"  Time range: {chunk['start_time']:.2f}s - {chunk['end_time']:.2f}s")
        
        if chunk['first_word']:
            print(f"  First word: '{chunk['first_word'].get('word', '')}' at {chunk['first_word'].get('start', 0):.2f}s")
        if chunk['last_word']:
            print(f"  Last word: '{chunk['last_word'].get('word', '')}' at {chunk['last_word'].get('end', 0):.2f}s")
        
        # Check for gap with next chunk
        if i < len(all_chunks) - 1:
            next_chunk = all_chunks[i+1]
            gap = next_chunk['start_time'] - chunk['end_time']
            print(f"  Gap to next chunk: {gap:.2f}s")
            
            # Check for timestamp overlap issues
            if gap < 0:
                print(f"  ⚠️ WARNING: Overlap with next chunk: {abs(gap):.2f}s")
            elif gap > 1.0:
                print(f"  ⚠️ WARNING: Large gap to next chunk: {gap:.2f}s")
        
        print("-" * 80)

def main():
    parser = argparse.ArgumentParser(description="Debug Groq chunking by forcing chunking and saving JSON files")
    parser.add_argument("--file", required=True, help="Audio/video file to transcribe")
    parser.add_argument("--chunk-size", type=int, default=30, help="Chunk size in seconds (default: 30)")
    args = parser.parse_args()
    
    # Create output directory for debug files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_dir = os.path.join("debug_chunks", f"groq_debug_{timestamp}")
    os.makedirs(debug_dir, exist_ok=True)
    
    print(f"Debugging Groq chunking for file: {args.file}")
    print(f"Chunk size: {args.chunk_size} seconds")
    print(f"Debug files will be saved to: {debug_dir}")
    
    # Run groq transcription with forced chunking and debug options
    cmd = [
        "uv", "run", "transcribe.py", 
        "--api", "groq",
        "--model", "whisper-large-v3",
        "--file", args.file,
        "--force",
        "--force-chunking",
        f"--chunk-size={args.chunk_size}",
        "--save-chunks",
        "--save-cleaned-json",
        "-v", "-d"
    ]
    
    print("\nRunning command:")
    print(" ".join(cmd))
    
    # Run the command and capture output
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    # Save command output for debugging
    output_log = os.path.join(debug_dir, "command_output.log")
    with open(output_log, 'w', encoding='utf-8') as f:
        f.write(process.stdout)
    
    # Find all chunk files created
    chunk_files = []
    base_name = os.path.splitext(args.file)[0]
    
    # Standard chunk pattern
    for root, dirs, files in os.walk(os.path.dirname(args.file)):
        for file in files:
            if file.endswith("_groq_chunk.json"):
                chunk_files.append(os.path.join(root, file))
    
    # Temp directory chunk pattern
    for root, dirs, files in os.walk(os.getenv('TEMP', '/tmp')):
        for file in files:
            if file.endswith("_groq.json"):
                chunk_files.append(os.path.join(root, file))
    
    # Analyze the chunks
    if chunk_files:
        print(f"\nFound {len(chunk_files)} chunk files:")
        for file in chunk_files:
            print(f"  {file}")
        
        # Copy chunks to debug directory
        for i, file in enumerate(chunk_files):
            file_name = f"chunk_{i+1}_{os.path.basename(file)}"
            dest_path = os.path.join(debug_dir, file_name)
            
            with open(file, 'r', encoding='utf-8') as f_in:
                with open(dest_path, 'w', encoding='utf-8') as f_out:
                    f_out.write(f_in.read())
            print(f"Copied {file} to {dest_path}")
        
        # Analyze chunks
        analyze_chunks(chunk_files)
    else:
        print("No chunk files found!")
    
    # Check final output
    final_json = f"{base_name}_groq.json"
    if os.path.exists(final_json):
        print(f"\nFinal merged output: {final_json}")
        
        # Copy final output to debug directory
        final_debug_path = os.path.join(debug_dir, "final_output.json")
        with open(final_json, 'r', encoding='utf-8') as f_in:
            with open(final_debug_path, 'w', encoding='utf-8') as f_out:
                f_out.write(f_in.read())
    else:
        print("\nNo final output file found!")
    
    print("\nDebug completed. All files saved to:", debug_dir)

if __name__ == "__main__":
    main() 