#!/usr/bin/env python3
"""
Build script for creating a standalone executable using PyInstaller.

Run this script to build the executable:
    python build.py

The executable will be placed in the ./dist directory.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def run_command(cmd, check=True):
    """Run a command with improved error handling."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        if check:
            sys.exit(1)
    main() 