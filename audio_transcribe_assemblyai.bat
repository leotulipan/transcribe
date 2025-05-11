@echo off

set "filename=%~1"
cd /d "%~dp1"

REM Call the unified transcribe.py script with AssemblyAI API
uv run --link-mode="copy" "%~dp0\transcribe.py" --file "%filename%" --api assemblyai --debug

pause