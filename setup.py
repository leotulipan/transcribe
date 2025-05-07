#!/usr/bin/env python3
"""
Setup script for the audio-transcribe package.
"""
from setuptools import setup, find_packages

# install with 
#   pip install -e .
# not yet working. look more at https://github.com/simonw/llm/blob/main/llm/cli.py

setup(
    name="audio-transcribe",
    version="0.1.0",
    description="Unified audio transcription tool supporting multiple APIs",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "click",
        "loguru",
        "pydub",
        "python-dotenv",
        "requests",
        "assemblyai",
        "groq",
    ],
    entry_points={
        "console_scripts": [
            "transcribe=audio_transcribe.cli:main",
        ],
    },
    python_requires=">=3.8",
)