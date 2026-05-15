@echo off

transcribe -l DE -a elevenlabs -o srt -c 55 -v "%~1"

REM transcribe -a elevenlabs -o srt -c 55 -v "%~1"

REM transcribe -l DE -a assemblyai -o text -m universal-3-pro --verbose 


pause 