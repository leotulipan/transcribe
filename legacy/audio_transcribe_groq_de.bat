@echo off

set "filename=%~1"
cd /d "%~dp1"

REM Call the unified transcribe.py script with Groq API and German language
uv run --link-mode=copy "%~dp0\transcribe.py" --file "%filename%" --api groq --language de --debug --verbose --save-cleaned-json

pause