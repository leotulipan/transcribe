@echo off
echo Debugging Groq chunking timestamps
echo ======================================

rem Fix for SSL certificate issues on Windows
set PYTHONHTTPSVERIFY=0
set REQUESTS_CA_BUNDLE=

rem Use a longer file to test chunking
set TEST_FILE="%1"

if "%TEST_FILE%"=="" (
  echo Error: Please provide a file path as an argument
  echo Usage: debug_groq_chunking.bat [path\to\audio_file]
  echo.
  echo Example: debug_groq_chunking.bat "C:\path\to\longer_audio.mp3"
  goto :end
)

rem Run the debug script with uv
uv run debug_groq_chunking.py --file %TEST_FILE% --chunk-size 20

:end
echo.
echo Debug completed. Press any key to exit...
pause > nul 