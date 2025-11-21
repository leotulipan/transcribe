@echo off
REM Template batch file for transcribing with Groq API (German)
REM Usage: Drag and drop an audio/video file onto this file, or run: transcribe_groq_de.bat "path\to\file.mp4"

set "filename=%~1"
if "%filename%"=="" (
    echo Usage: Drag and drop a file onto this script, or run: %~nx0 "path\to\file.mp4"
    pause
    exit /b 1
)

cd /d "%~dp1"

REM Call transcribe.exe with Groq API and German language
"%~dp0transcribe.exe" --file "%filename%" --api groq --language de --verbose

pause

