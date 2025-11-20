@echo off
setlocal EnableDelayedExpansion

REM Check if URL is provided
if "%~1"=="" (
    set /p "URL=Enter YouTube URL: "
) else (
    set "URL=%~1"
)

REM Check if URL is still empty
if "!URL!"=="" (
    echo No URL provided. Exiting.
    pause
    exit /b 1
)

REM Set the output filename
set "OUTPUT_FILE=transcript.txt"

REM Function to download subtitles
:download_subs
set "sub_lang=%~1"
set "sub_type=%~2"

REM Construct the yt-dlp command
set "yt_dlp_command=uvx yt-dlp --skip-download --write-%sub_type% --sub-lang %sub_lang% --sub-format srt --output "%%(title)s.%%(ext)s" "!URL!""

REM Execute and filter, redirecting stderr to stdout
%yt_dlp_command% 2>&1 | findstr /i /v /c:"WARNING:" > temp_output.txt

REM Get the base filename (without extension) for checking SRT existence
for /f "tokens=*" %%a in ('uvx yt-dlp --get-filename --output "%%(title)s.%%(ext)s" "!URL!"') do (
    set "base_filename=%%~na"
)
set "srt_filename=!base_filename!.srt"


if exist "!srt_filename!" (
    REM Subtitles downloaded. Move and rename.
    echo Subtitles found (%sub_lang%).
    move /y "!srt_filename!" "!OUTPUT_FILE!"
    del temp_output.txt
    goto :success
) else (
    REM Check for No Subtitle message
    findstr /i /c:"There are no subtitles" temp_output.txt >nul
    if !errorlevel! equ 0 (
        echo No subtitles found for language: %sub_lang%.
    ) else (
        REM Other error. Display it.
        echo.
        echo Error downloading subtitles:
        type temp_output.txt
        echo.
        pause
        del temp_output.txt
        exit /b 1
    )
    del temp_output.txt
    goto :eof
)


REM --- Main Script Flow ---

REM 1. Try original subtitles (any language).
call :download_subs original subs
if !errorlevel! equ 0 goto :success

REM 2. Try auto-generated English subtitles.
call :download_subs en auto-subs
if !errorlevel! equ 0 goto :success

REM 3. Try "en-orig" subtitles.
call :download_subs en-orig subs
if !errorlevel! equ 0 goto :success


REM If no subtitles found, list available.
echo No suitable subtitles found. Listing all available subtitles:
uvx yt-dlp --list-subs "!URL!"
pause
exit /b 1

:success
echo Transcript saved to !OUTPUT_FILE!
pause
exit /b 0

endlocal