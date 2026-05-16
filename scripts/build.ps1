param(
    [string]$Version = "dev",
    [string]$OutDir = "bin"
)
$ErrorActionPreference = "Stop"
if (-not $Version -or $Version -eq "dev") {
    try { $Version = (git describe --tags --always 2>$null) } catch {}
    if (-not $Version) { $Version = "dev" }
}
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }

Write-Host "Building $OutDir/transcribe.exe (version=$Version)"
go build -ldflags "-X main.version=$Version" -o "$OutDir/transcribe.exe" ./cmd/transcribe

Write-Host "Building $OutDir/transcribe-gui.exe (version=$Version, no console)"
go build -ldflags "-X main.version=$Version -H windowsgui" -o "$OutDir/transcribe-gui.exe" ./cmd/transcribe-gui
