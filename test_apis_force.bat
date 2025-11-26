@echo off
REM Test all APIs with force flag to create new JSONs in test folder

echo Testing all APIs with force flag...
echo.

echo [1/4] Testing AssemblyAI...
uv run transcribe test\audio-test.mkv --api assemblyai --force --save-cleaned-json
echo.

echo [2/4] Testing ElevenLabs...
uv run transcribe test\audio-test.mkv --api elevenlabs --force --save-cleaned-json
echo.

echo [3/4] Testing Groq...
uv run transcribe test\audio-test.mkv --api groq --force --save-cleaned-json
echo.

echo [4/4] Testing OpenAI...
uv run transcribe test\audio-test.mkv --api openai --force --save-cleaned-json
echo.

echo Done! All APIs tested.
pause

