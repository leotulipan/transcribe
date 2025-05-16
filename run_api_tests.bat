@echo off
echo Running API tests for transcribe.py
echo ======================================

rem Fix for SSL certificate issues on Windows
set PYTHONHTTPSVERIFY=0
set REQUESTS_CA_BUNDLE=

rem Run the test script with uv
uv run test_api_outputs.py

echo.
echo Test completed. Press any key to exit...
pause > nul 