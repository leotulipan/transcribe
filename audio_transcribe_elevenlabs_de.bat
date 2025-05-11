@echo off

set "filename=%~1"
cd /d "%~dp1"

REM Call the unified transcribe.py script with ElevenLabs API and German language
uv run --link-mode=copy "%~dp0\transcribe.py" --file "%filename%" --api elevenlabs --language de --verbose --debug --show-pauses --davinci-srt --padding-start 50 --silent-portions 350

pause 