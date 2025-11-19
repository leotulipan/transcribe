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
import platform
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
        return e

def install_dependencies():
    """Install all required dependencies using UV."""
    print("Installing required dependencies...")
    
    # Install main dependencies with UV
    dependencies = [
        "pyinstaller",
        "click",
        "loguru",
        "pydub",
        "python-dotenv",
        "requests",
        "assemblyai",
        "groq",
        "openai",
        "questionary",
        "rich"
    ]
    
    # Install dependencies all at once for better performance
    cmd = ["uv", "pip", "install"] + dependencies
    run_command(cmd)
    
    # Install the current package in development mode
    print("Installing current package...")
    run_command(["uv", "pip", "install", "-e", "."])

def main():
    """Build the executable using PyInstaller."""
    print("Building audio-transcribe executable...")
    
    # Install dependencies
    install_dependencies()
    
    # Import loguru after installing dependencies
    try:
        import loguru
        loguru_path = os.path.dirname(loguru.__file__)
    except ImportError:
        print("Error: loguru module not found after installation")
        sys.exit(1)
    
    # Create the dist directory if it doesn't exist
    os.makedirs("dist", exist_ok=True)
    
    # Get the absolute path to the CLI entry point
    cli_path = os.path.abspath("audio_transcribe/cli.py")
    if not os.path.exists(cli_path):
        print(f"Error: Entry point not found at {cli_path}")
        sys.exit(1)
    
    # Clean up any previous PyInstaller artifacts
    for artifact in ["build", "audio_transcribe.spec"]:
        if os.path.exists(artifact):
            if os.path.isdir(artifact):
                shutil.rmtree(artifact)
            else:
                os.remove(artifact)
    
    # Determine the correct separator for --add-data based on platform
    separator = ";" if os.name == "nt" else ":"
    
    # Determine output name based on platform
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    output_name = f"transcribe-{system}-{machine}"
    
    # Run PyInstaller directly with all needed options
    print(f"Building executable {output_name} with PyInstaller...")
    run_command([
        "pyinstaller",
        f"--name={output_name}",
        "--onefile",
        "--noconfirm",
        "--clean",
        "--add-data", f"{loguru_path}{separator}loguru",
        f"--workpath={os.path.abspath('build')}",
        f"--distpath={os.path.abspath('dist')}",
        f"--specpath={os.path.abspath('.')}",
        "--hidden-import=assemblyai",
        "--hidden-import=groq",
        "--hidden-import=openai",
        "--hidden-import=requests", 
        "--hidden-import=python-dotenv",
        "--hidden-import=pydub",
        "--hidden-import=questionary",
        "--hidden-import=rich",
        cli_path
    ])
    
    # Verify the executable was created
    exe_extension = ".exe" if os.name == "nt" else ""
    exe_path = os.path.abspath(f"dist/{output_name}{exe_extension}")
    if os.path.exists(exe_path):
        print(f"\nBuild successful! Executable created at: {exe_path}")
        print("\nYou can now run the 'transcribe' command using the executable:")
        print(f"\n  {exe_path} --help")
    else:
        print("\nBuild failed: Executable not found!")
        sys.exit(1)

if __name__ == "__main__":
    main()