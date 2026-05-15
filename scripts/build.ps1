param(
    [string]$Version = "dev",
    [string]$Out = "bin/transcribe.exe"
)
$ErrorActionPreference = "Stop"
if (-not $Version -or $Version -eq "dev") {
    try { $Version = (git describe --tags --always 2>$null) } catch {}
    if (-not $Version) { $Version = "dev" }
}
Write-Host "Building $Out (version=$Version)"
go build -ldflags "-X main.version=$Version" -o $Out ./cmd/transcribe
