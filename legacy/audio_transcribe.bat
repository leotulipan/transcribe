@echo off

set "filename=%~1"
cd /d "%~dp1"

python C:\Users\leo\OneDrive\_2_Areas\Scripts.Transcribe\audio_transcribe.py --file "%filename%" --keep --debug
pause