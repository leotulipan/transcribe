import os
import subprocess
import time
from datetime import datetime

# APIs to test with their default models
API_CONFIG = {
    "assemblyai": {"model": "best"},
    "elevenlabs": {"model": "scribe_v1"},
    "groq": {"model": "whisper-large-v3"},
    "openai": {"model": "whisper-1"}  # This was missing
}
TEST_FILE = "./test/audio-test.mkv"

def get_file_info(filepath):
    """Get file existence and modified time if exists"""
    if os.path.exists(filepath):
        mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
        return True, mod_time
    return False, None

def main():
    results = {}
    error_logs = {}
    
    # Get baseline file info before running tests
    base_name = os.path.splitext(TEST_FILE)[0]
    baseline = {}
    
    for api in API_CONFIG.keys():
        json_path = f"{base_name}_{api}.json"
        srt_path = f"{base_name}.srt"
        txt_path = f"{base_name}.txt"
        
        baseline[api] = {
            "json": get_file_info(json_path),
            "srt": get_file_info(srt_path),
            "txt": get_file_info(txt_path),
        }
    
    # Run tests for each API
    for api, config in API_CONFIG.items():
        print(f"\n{'='*50}")
        print(f"Testing API: {api}")
        print(f"{'='*50}")
        
        # Run transcribe.py using uv with the model parameter
        cmd = [
            "uv", "run", "transcribe.py", 
            "--api", api, 
            "--force", 
            "--file", TEST_FILE,
            "--model", config["model"],
            "-v", "-d"
        ]
        print(f"Running: {' '.join(cmd)}")
        
        # Run the command and capture output
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        # Store error output for review
        if process.returncode != 0 or "ERROR" in process.stdout:
            error_logs[api] = process.stdout
            print(f"Command failed with return code: {process.returncode}")
        
        # Check results - we still check for files even if API call failed
        json_path = f"{base_name}_{api}.json"
        srt_path = f"{base_name}.srt"
        txt_path = f"{base_name}.txt"
        
        # Get current file info
        json_exists, json_time = get_file_info(json_path)
        srt_exists, srt_time = get_file_info(srt_path)
        txt_exists, txt_time = get_file_info(txt_path)
        
        # Compare with baseline
        json_updated = (not baseline[api]["json"][0]) or (json_time > baseline[api]["json"][1]) if json_exists else False
        srt_updated = (not baseline[api]["srt"][0]) or (srt_time > baseline[api]["srt"][1]) if srt_exists else False
        txt_updated = (not baseline[api]["txt"][0]) or (txt_time > baseline[api]["txt"][1]) if txt_exists else False
        
        results[api] = {
            "json_created": json_exists,
            "json_updated": json_updated,
            "srt_created": srt_exists,
            "srt_updated": srt_updated,
            "txt_created": txt_exists,
            "txt_updated": txt_updated,
            "api_error": api in error_logs
        }
        
        # Update baseline for next API
        baseline[api]["json"] = (json_exists, json_time)
        baseline[api]["srt"] = (srt_exists, srt_time)
        baseline[api]["txt"] = (txt_exists, txt_time)
    
    # Print summary
    print("\n\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    
    all_passed = True
    for api, result in results.items():
        print(f"\nAPI: {api}")
        print(f"  JSON file created: {'✅' if result['json_created'] else '❌'}")
        print(f"  JSON file updated: {'✅' if result['json_updated'] else '❌'}")
        print(f"  SRT file created: {'✅' if result['srt_created'] else '❌'}")
        print(f"  SRT file updated: {'✅' if result['srt_updated'] else '❌'}")
        print(f"  TXT file created: {'✅' if result['txt_created'] else '❌'}")
        print(f"  TXT file updated: {'✅' if result['txt_updated'] else '❌'}")
        print(f"  API Error occurred: {'❌' if result['api_error'] else '✅'}")
        
        # We only fail the test if JSON file wasn't created/updated
        if not (result['json_created'] and result['json_updated']):
            all_passed = False
    
    # Print error details if requested
    if error_logs:
        print("\nERROR DETAILS:")
        for api, log in error_logs.items():
            print(f"\nAPI: {api}")
            # Extract and print just the error messages
            for line in log.splitlines():
                if "ERROR" in line or "WARNING" in line:
                    print(f"  {line.strip()}")
    
    print("\n" + "="*80)
    print(f"OVERALL TEST STATUS: {'✅ PASSED' if all_passed else '❌ FAILED'}")
    print("="*80)

if __name__ == "__main__":
    main() 