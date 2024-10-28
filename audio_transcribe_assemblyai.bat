@echo off

set "filename=%~1"
cd /d "%~dp1"

REM python "%~dp0\audio_transcribe_assemblyai.py" --file "%filename%" --debug
uv run --link-mode="copy" "%~dp0\audio_transcribe_assemblyai.py" --file "%filename%" --debug
pause