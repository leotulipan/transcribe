param(
    [string]$Version = "dev",
    [string]$OutDir = "bin",
    [switch]$Install
)
$ErrorActionPreference = "Stop"

# The GUI (and the CLI, which imports the gui package) needs cgo for Fyne's GL
# driver. Enable it and, if no C compiler is on PATH, fall back to a standard
# MSYS2 ucrt64 toolchain when one is present.
$env:CGO_ENABLED = "1"
if (-not (Get-Command gcc -ErrorAction SilentlyContinue)) {
    $ucrt = "C:\msys64\ucrt64\bin"
    if (Test-Path (Join-Path $ucrt "gcc.exe")) {
        $env:Path = "$ucrt;$env:Path"
    }
}

# Derive version from git describe when -Version is "dev" or empty. We expect
# annotated tags of the form vMAJOR.MINOR.PATCH (semver 2.0.0). git describe
# will append "-<n>-g<sha>" for commits after a tag, which is still acceptable.
if (-not $Version -or $Version -eq "dev") {
    try { $Version = (git describe --tags --always 2>$null) } catch {}
    if (-not $Version) { $Version = "dev" }
}

# Soft validation: warn if the version string doesn't look like vMAJOR.MINOR.PATCH
# or a clean "dev"/"dev-..." marker. Doesn't fail the build — lets you produce
# ad-hoc smoke builds.
$semver = '^v\d+\.\d+\.\d+(-[0-9A-Za-z\-\.]+)?(\+[0-9A-Za-z\-\.]+)?$'
$gitDesc = '^v\d+\.\d+\.\d+-\d+-g[0-9a-f]+$'
$devTag  = '^dev(-[A-Za-z0-9\.\-_]+)?$'
if ($Version -notmatch $semver -and $Version -notmatch $gitDesc -and $Version -notmatch $devTag) {
    Write-Warning "Version '$Version' is not vMAJOR.MINOR.PATCH semver. Tag a release with 'git tag -a vX.Y.Z -m ...' for a clean version."
}

if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }

Write-Host "Building $OutDir/transcribe.exe (version=$Version)"
go build -ldflags "-X main.version=$Version" -o "$OutDir/transcribe.exe" ./cmd/transcribe
if ($LASTEXITCODE -ne 0) { throw "go build failed for transcribe.exe (exit $LASTEXITCODE)" }

Write-Host "Building $OutDir/transcribe-gui.exe (version=$Version, no console)"
go build -ldflags "-X main.version=$Version -H windowsgui" -o "$OutDir/transcribe-gui.exe" ./cmd/transcribe-gui
if ($LASTEXITCODE -ne 0) { throw "go build failed for transcribe-gui.exe (exit $LASTEXITCODE)" }

# Optionally copy the freshly built binaries into the local install locations
# and verify them, so `build.ps1 -Install` is a one-shot build+install+verify.
if ($Install) {
    & "$PSScriptRoot/install-local.ps1" -SourceDir $OutDir
}
