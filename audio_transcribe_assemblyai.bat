@echo off

set "filename=%~1"
cd /d "%~dp1"

python "%~dp0\audio_transcribe_assemblyai.py" --file "%filename%" --debug
pause