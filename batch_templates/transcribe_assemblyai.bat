@echo off
REM Template batch file for transcribing with AssemblyAI API
REM Usage: Drag and drop an audio/video file onto this file, or run: transcribe_assemblyai.bat "path\to\file.mp4"

set "filename=%~1"
if "%filename%"=="" (
    echo Usage: Drag and drop a file onto this script, or run: %~nx0 "path\to\file.mp4"
    pause
    exit /b 1
)

cd /d "%~dp1"

REM Call transcribe.exe with AssemblyAI API
"%~dp0transcribe.exe" --file "%filename%" --api assemblyai --verbose

pause

