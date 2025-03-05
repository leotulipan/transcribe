@echo off

set "filename=%~1"
cd /d "%~dp1"

uv run --link-mode=copy C:\Users\leona\OneDrive\_2_Areas\Scripts.Transcribe\groq_audio_chunking_adapted.py -l de "%filename%"

pause