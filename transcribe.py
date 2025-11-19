#!/usr/bin/env python3
# /// script
# dependencies = [
#   "click",
#   "loguru",
#   "pydub",
#   "pathlib",
#   "python-dotenv",
#   "requests",
#   "assemblyai",
#   "groq",
#   "openai",
# ]
# ///

"""
Unified Audio Transcription Tool - Wrapper Script

This script wraps the audio_transcribe package CLI.
"""

import sys
import os

# Add current directory to path so we can import the package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from audio_transcribe.cli import main

if __name__ == "__main__":
    main()