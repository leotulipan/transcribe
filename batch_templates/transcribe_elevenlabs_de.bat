@echo off
REM Template batch file for transcribing with ElevenLabs API (German)
REM Usage: Drag and drop an audio/video file onto this file, or run: transcribe_elevenlabs_de.bat "path\to\file.mp4"

set "filename=%~1"
if "%filename%"=="" (
    echo Usage: Drag and drop a file onto this script, or run: %~nx0 "path\to\file.mp4"
    pause
    exit /b 1
)

cd /d "%~dp1"

REM Call transcribe.exe with ElevenLabs API, German language, and DaVinci Resolve optimized output
"%~dp0transcribe.exe" --file "%filename%" --api elevenlabs --language de --verbose --davinci-srt --show-pauses --silent-portions 350

pause

