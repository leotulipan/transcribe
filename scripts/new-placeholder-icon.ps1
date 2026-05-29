# scripts/new-placeholder-icon.ps1
#
# Generates a placeholder app icon end-to-end using ImageMagick so the build
# never blocks on a manually-supplied asset. Run automatically by
# scripts/build-installer.ps1 when assets/icon.ico is missing; also safe to
# invoke directly.
#
# Output:
#   assets/icon-1024.png  (1024x1024 source PNG, transparent background)
#   assets/icon.ico       (multi-resolution: 16, 24, 32, 48, 64, 128, 256)
#
# Replace these files with a real branded icon before public release. The
# placeholder uses a bright blue rounded square with a bold "T" so it is
# instantly recognizable as not-the-final-icon.

[CmdletBinding()]
param(
    [string]$RepoRoot = (Resolve-Path "$PSScriptRoot\..").Path
)

$ErrorActionPreference = "Stop"

$assetsDir = Join-Path $RepoRoot "assets"
if (-not (Test-Path $assetsDir)) {
    New-Item -ItemType Directory -Path $assetsDir | Out-Null
}

$pngPath = Join-Path $assetsDir "icon-1024.png"
$icoPath = Join-Path $assetsDir "icon.ico"

# Locate ImageMagick. Prefer 'magick' (v7+), fall back to legacy 'convert'.
$magick = Get-Command magick -ErrorAction SilentlyContinue
if (-not $magick) {
    throw "ImageMagick (magick.exe) not found on PATH. Install with: winget install ImageMagick.ImageMagick"
}

Write-Host "Rendering 1024x1024 placeholder PNG -> $pngPath"
& $magick.Source `
    -size 1024x1024 xc:none `
    -fill "#3B82F6" `
    -draw "roundrectangle 32,32 992,992 128,128" `
    -fill white `
    -gravity center `
    -pointsize 720 `
    -font "Arial-Bold" `
    -annotate +0+40 "T" `
    $pngPath
if ($LASTEXITCODE -ne 0) { throw "ImageMagick failed to render placeholder PNG (exit $LASTEXITCODE)" }

Write-Host "Building multi-resolution ICO -> $icoPath"
& $magick.Source $pngPath `
    -define icon:auto-resize=256,128,64,48,32,24,16 `
    $icoPath
if ($LASTEXITCODE -ne 0) { throw "ImageMagick failed to build ICO (exit $LASTEXITCODE)" }

$pngSize = (Get-Item $pngPath).Length
$icoSize = (Get-Item $icoPath).Length
Write-Host ""
Write-Host "Placeholder icon ready:" -ForegroundColor Green
Write-Host ("  {0}  ({1:N0} bytes)" -f $pngPath, $pngSize)
Write-Host ("  {0}  ({1:N0} bytes)" -f $icoPath, $icoSize)
Write-Host ""
Write-Host "Replace these files with a real icon before public release." -ForegroundColor Yellow
