@echo off

set "filename=%~1"
cd /d "%~dp1"

uv run --link-mode=copy C:\Users\leona\OneDrive\_2_Areas\Scripts.Transcribe\audio_transcribe_elevenlabs.py -d -v -s -D --padding 50 -p 350 "%filename%"

pause 