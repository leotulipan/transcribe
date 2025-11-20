@echo off

set "filename=%~1"
cd /d "%~dp1"

REM Call the unified transcribe.py script with AssemblyAI API
REM uv run --link-mode="copy" "%~dp0\transcribe.py" --file "%filename%" --api assemblyai --debug
uv run "%~dp0\transcribe.py" --api assemblyai --debug --verbose -c 42 --file "%filename%"

pause