# scripts/check-installer-prereqs.ps1
#
# Preflight check for the Windows installer build pipeline. Runs the same
# probes that scripts/build-installer.ps1 uses but does not build anything.
# Useful as a doctor command for new contributors and as a CI smoke test
# before the (much longer) full build.
#
# Exit code 0 if everything is present, non-zero if any check fails.
#
# Usage:
#   pwsh scripts/check-installer-prereqs.ps1

[CmdletBinding()]
param()

$ErrorActionPreference = "Continue"

$failures = @()

function Check {
    param([string]$Label, [scriptblock]$Probe, [string]$Hint)
    Write-Host -NoNewline ("  {0,-30}" -f $Label)
    try {
        $result = & $Probe
        if ($result) {
            Write-Host "OK  $result" -ForegroundColor Green
        } else {
            Write-Host "MISSING" -ForegroundColor Red
            $script:failures += "{0}: {1}" -f $Label, $Hint
        }
    } catch {
        Write-Host "MISSING ($($_.Exception.Message))" -ForegroundColor Red
        $script:failures += "{0}: {1}" -f $Label, $Hint
    }
}

Write-Host ""
Write-Host "=== Windows installer prerequisites ===" -ForegroundColor Cyan
Write-Host ""

Check "Go toolchain" {
    $g = Get-Command go -ErrorAction SilentlyContinue
    if ($g) { (& go version) } else { $null }
} "Install Go from https://go.dev/dl/"

Check "C compiler (CGo)" {
    $g = Get-Command gcc -ErrorAction SilentlyContinue
    if ($g) { return $g.Source }
    $candidates = @(
        'C:\msys64\ucrt64\bin\gcc.exe',
        'C:\msys64\mingw64\bin\gcc.exe',
        'C:\TDM-GCC-64\bin\gcc.exe',
        'C:\mingw64\bin\gcc.exe',
        "$env:USERPROFILE\scoop\apps\mingw\current\bin\gcc.exe",
        "$env:LOCALAPPDATA\Programs\mingw64\bin\gcc.exe"
    )
    foreach ($c in $candidates) { if ($c -and (Test-Path $c)) { return $c } }
    $null
} "Install MSYS2 (winget install MSYS2.MSYS2) then 'pacman -S mingw-w64-ucrt-x86_64-gcc' inside the UCRT64 shell"

Check "goversioninfo" {
    $goBin = (& go env GOBIN); if (-not $goBin) { $goBin = Join-Path (& go env GOPATH) "bin" }
    $gv = Join-Path $goBin "goversioninfo.exe"
    if (Test-Path $gv) { $gv } else { $null }
} "Auto-installed on first build; or run: go install github.com/josephspurrier/goversioninfo/cmd/goversioninfo@latest"

Check "Inno Setup 6 (ISCC.exe)" {
    $cmd = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $candidates = @(
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "${env:ProgramFiles}\Inno Setup 6\ISCC.exe",
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
    )
    foreach ($c in $candidates) { if ($c -and (Test-Path $c)) { return $c } }
    $null
} "winget install JRSoftware.InnoSetup"

Check "ImageMagick (magick.exe)" {
    $cmd = Get-Command magick -ErrorAction SilentlyContinue
    if ($cmd) { $cmd.Source } else { $null }
} "Only required when assets/icon.ico is missing. Install: winget install ImageMagick.ImageMagick"

Check "git" {
    $cmd = Get-Command git -ErrorAction SilentlyContinue
    if ($cmd) { (& git --version) } else { $null }
} "Install Git for Windows. Required for version detection via 'git describe --tags'."

Write-Host ""
if ($failures.Count -eq 0) {
    Write-Host "All prerequisites present. Run scripts/build-installer.ps1." -ForegroundColor Green
    exit 0
} else {
    Write-Host "Missing prerequisites:" -ForegroundColor Red
    foreach ($f in $failures) { Write-Host "  - $f" }
    exit 1
}
