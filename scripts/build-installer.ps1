# scripts/build-installer.ps1
#
# Full release orchestrator for the Audio Transcribe Windows installer.
# Runs identically on a local Win11 dev box and on a GitHub Actions
# windows-latest runner.
#
# Flow:
#   1. Resolve version (arg, or git describe, with fallback)
#   2. Verify prerequisites (go, goversioninfo, ISCC; magick only if icon missing)
#   3. Bootstrap placeholder icon if assets/icon.ico is absent
#   4. Render per-binary versioninfo.json from installer/versioninfo.tmpl.json
#   5. go generate -> resource.syso for each cmd
#   6. go build CLI + GUI into dist/
#   7. Compile installer with Inno Setup -> dist/transcribe-setup-v<ver>.exe
#   8. Compute SHA256 sidecar
#   9. Print summary
#
# Usage:
#   pwsh scripts/build-installer.ps1                       # version from git
#   pwsh scripts/build-installer.ps1 -Version 0.11.0       # explicit
#   pwsh scripts/build-installer.ps1 -SkipInstaller        # just build binaries

[CmdletBinding()]
param(
    [string]$Version,
    [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"

# ----- Paths -----------------------------------------------------------------
$repoRoot   = (Resolve-Path "$PSScriptRoot\..").Path
$distDir    = Join-Path $repoRoot "dist"
$assetsDir  = Join-Path $repoRoot "assets"
$iconPath   = Join-Path $assetsDir "icon.ico"
$installer  = Join-Path $repoRoot "installer"
$tmplPath   = Join-Path $installer "versioninfo.tmpl.json"
$issPath    = Join-Path $installer "transcribe.iss"
$cliDir     = Join-Path $repoRoot "cmd\transcribe"
$guiDir     = Join-Path $repoRoot "cmd\transcribe-gui"

Set-Location $repoRoot

# ----- 1. Version ------------------------------------------------------------
function Resolve-Version {
    param([string]$Override)
    if ($Override) { return $Override.TrimStart('v') }
    try {
        $tag = (git describe --tags --abbrev=0 2>$null)
        if ($LASTEXITCODE -eq 0 -and $tag) {
            return $tag.Trim().TrimStart('v')
        }
    } catch {}
    Write-Warning "No git tag found; using fallback version 0.0.0-dev"
    return "0.0.0-dev"
}

$version = Resolve-Version -Override $Version

# Parse Major/Minor/Patch for the embedded FixedFileInfo block. Anything trailing
# (e.g. "0.10.0-dev" or "0.10.0-5-gabcdef") is acceptable for the string fields
# but we need three integers for the binary version record.
if ($version -match '^(\d+)\.(\d+)\.(\d+)') {
    $vMajor = [int]$Matches[1]
    $vMinor = [int]$Matches[2]
    $vPatch = [int]$Matches[3]
} else {
    Write-Warning "Version '$version' does not start with N.N.N; embedding 0.0.0 in binary FixedFileInfo"
    $vMajor = 0; $vMinor = 0; $vPatch = 0
}

Write-Host ""
Write-Host "=== Audio Transcribe installer build ===" -ForegroundColor Cyan
Write-Host ("Version:  {0}  ({1}.{2}.{3})" -f $version, $vMajor, $vMinor, $vPatch)
Write-Host ("Repo:     {0}" -f $repoRoot)
Write-Host ""

# ----- 2. Prereqs ------------------------------------------------------------
function Require-Tool {
    param([string]$Name, [string]$HintIfMissing)
    $tool = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $tool) { throw "Required tool '$Name' not found on PATH. $HintIfMissing" }
    return $tool.Source
}

$null = Require-Tool -Name "go" -HintIfMissing "Install Go from https://go.dev/dl/"

# CGo + C compiler. Fyne pulls in OpenGL bindings (github.com/go-gl/gl) that
# require CGO_ENABLED=1 and a C compiler on PATH. On a clean Windows dev box
# the compiler typically comes from MSYS2 (UCRT64 toolchain). Probe known
# locations and inject into PATH for this process so `go build` succeeds.
if (-not (Get-Command gcc -ErrorAction SilentlyContinue)) {
    $gccCandidates = @(
        'C:\msys64\ucrt64\bin\gcc.exe',
        'C:\msys64\mingw64\bin\gcc.exe',
        'C:\TDM-GCC-64\bin\gcc.exe',
        'C:\mingw64\bin\gcc.exe',
        "$env:USERPROFILE\scoop\apps\mingw\current\bin\gcc.exe",
        "$env:LOCALAPPDATA\Programs\mingw64\bin\gcc.exe"
    )
    $gccPath = $null
    foreach ($c in $gccCandidates) {
        if ($c -and (Test-Path $c)) { $gccPath = $c; break }
    }
    if (-not $gccPath) {
        throw "C compiler (gcc) not found. Fyne requires CGo with a working C compiler. Install MSYS2 (winget install MSYS2.MSYS2) and the UCRT64 gcc package (pacman -S mingw-w64-ucrt-x86_64-gcc), or install TDM-GCC."
    }
    $gccDir = Split-Path -Parent $gccPath
    if (($env:PATH -split ';') -notcontains $gccDir) {
        $env:PATH = "$gccDir;$env:PATH"
    }
    Write-Host "Using C compiler at $gccPath"
}
$env:CGO_ENABLED = "1"


# goversioninfo: install on demand into GOPATH\bin so it's available going forward.
$goBin = (& go env GOBIN); if (-not $goBin) { $goBin = Join-Path (& go env GOPATH) "bin" }
$goversioninfo = Join-Path $goBin "goversioninfo.exe"
if (-not (Test-Path $goversioninfo)) {
    Write-Host "Installing goversioninfo into $goBin..."
    & go install github.com/josephspurrier/goversioninfo/cmd/goversioninfo@latest
    if ($LASTEXITCODE -ne 0) { throw "Failed to install goversioninfo (exit $LASTEXITCODE)" }
}
if (-not (Test-Path $goversioninfo)) { throw "goversioninfo not found at $goversioninfo after install" }
# Ensure goversioninfo is reachable from `go generate` (which doesn't inherit
# changes to the parent shell's PATH made later in this script).
if (($env:PATH -split ';') -notcontains $goBin) {
    $env:PATH = "$goBin;$env:PATH"
}

# ISCC (Inno Setup compiler): search PATH first, then the default install dir.
$iscc = $null
$isccCmd = Get-Command ISCC.exe -ErrorAction SilentlyContinue
if ($isccCmd) {
    $iscc = $isccCmd.Source
} else {
    $candidates = @(
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "${env:ProgramFiles}\Inno Setup 6\ISCC.exe",
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
    )
    foreach ($c in $candidates) {
        if ($c -and (Test-Path $c)) { $iscc = $c; break }
    }
}
if (-not $SkipInstaller -and -not $iscc) {
    throw "Inno Setup 6 not found in PATH or these locations:`n  $($candidates -join "`n  ")`nInstall with: winget install JRSoftware.InnoSetup"
}

# ----- 3. Icon bootstrap -----------------------------------------------------
if (-not (Test-Path $iconPath)) {
    Write-Host "assets/icon.ico missing - generating placeholder via ImageMagick..."
    & (Join-Path $PSScriptRoot "new-placeholder-icon.ps1")
    if (-not (Test-Path $iconPath)) { throw "Placeholder icon generator did not produce $iconPath" }
}

# ----- 4. Render versioninfo.json per binary ---------------------------------
function Render-VersionInfo {
    param(
        [string]$TargetDir,
        [string]$InternalName,
        [string]$OriginalName,
        [string]$Description
    )
    $content = Get-Content $tmplPath -Raw
    $content = $content.Replace('${VERSION_MAJOR}',   "$vMajor")
    $content = $content.Replace('${VERSION_MINOR}',   "$vMinor")
    $content = $content.Replace('${VERSION_PATCH}',   "$vPatch")
    $content = $content.Replace('${VERSION_STRING}',  "$version")
    $content = $content.Replace('${INTERNAL_NAME}',   "$InternalName")
    $content = $content.Replace('${ORIGINAL_NAME}',   "$OriginalName")
    $content = $content.Replace('${DESCRIPTION}',     "$Description")
    $outPath = Join-Path $TargetDir "versioninfo.json"
    Set-Content -Path $outPath -Value $content -Encoding UTF8 -NoNewline
    Write-Host "  wrote $outPath"
}

Write-Host "Rendering versioninfo.json files..."
Render-VersionInfo -TargetDir $cliDir -InternalName "transcribe"     -OriginalName "transcribe.exe"     -Description "Audio Transcribe (CLI)"
Render-VersionInfo -TargetDir $guiDir -InternalName "transcribe-gui" -OriginalName "transcribe-gui.exe" -Description "Audio Transcribe"

# ----- 5. go generate (-> resource.syso) -------------------------------------
Write-Host "Running 'go generate' to produce resource.syso files..."
& go generate ./cmd/transcribe ./cmd/transcribe-gui
if ($LASTEXITCODE -ne 0) { throw "go generate failed (exit $LASTEXITCODE)" }

foreach ($p in @("$cliDir\resource.syso", "$guiDir\resource.syso")) {
    if (-not (Test-Path $p)) { throw "Expected resource.syso not produced at $p" }
    Write-Host ("  {0}  ({1:N0} bytes)" -f $p, (Get-Item $p).Length)
}

# ----- 6. Build binaries -----------------------------------------------------
if (-not (Test-Path $distDir)) { New-Item -ItemType Directory -Path $distDir | Out-Null }

Write-Host "Building dist/transcribe.exe (CLI)..."
& go build -ldflags "-X main.version=$version -s -w" -o "$distDir\transcribe.exe" ./cmd/transcribe
if ($LASTEXITCODE -ne 0) { throw "CLI build failed (exit $LASTEXITCODE)" }

Write-Host "Building dist/transcribe-gui.exe (GUI, no console)..."
& go build -ldflags "-X main.version=$version -H windowsgui -s -w" -o "$distDir\transcribe-gui.exe" ./cmd/transcribe-gui
if ($LASTEXITCODE -ne 0) { throw "GUI build failed (exit $LASTEXITCODE)" }

# Quick sanity check: confirm the embedded ProductVersion matches.
foreach ($exe in @("$distDir\transcribe.exe", "$distDir\transcribe-gui.exe")) {
    $info = (Get-Item $exe).VersionInfo
    Write-Host ("  {0}  v{1}  ({2:N0} bytes)" -f $exe, $info.ProductVersion, (Get-Item $exe).Length)
}

if ($SkipInstaller) {
    Write-Host ""
    Write-Host "Skipping installer (-SkipInstaller). Binaries are in dist/." -ForegroundColor Yellow
    return
}

# ----- 7. Inno Setup ---------------------------------------------------------
Write-Host ""
Write-Host "Compiling installer with $iscc..."
& $iscc /Qp "/DAppVersion=$version" "/DSourceDir=$distDir" $issPath
if ($LASTEXITCODE -ne 0) { throw "Inno Setup compile failed (exit $LASTEXITCODE)" }

$setupExe = Join-Path $distDir "transcribe-setup-v$version.exe"
if (-not (Test-Path $setupExe)) { throw "Expected installer not produced at $setupExe" }

# ----- 8. SHA256 -------------------------------------------------------------
$hash = (Get-FileHash $setupExe -Algorithm SHA256).Hash.ToLower()
$sumPath = "$setupExe.sha256"
Set-Content -Path $sumPath -Value "$hash  $(Split-Path -Leaf $setupExe)" -Encoding ASCII -NoNewline
Write-Host "SHA256: $hash"

# ----- 9. Summary ------------------------------------------------------------
Write-Host ""
Write-Host "=== Build complete ===" -ForegroundColor Green
$setupItem = Get-Item $setupExe
Write-Host ("Installer:  {0}" -f $setupExe)
Write-Host ("Size:       {0:N0} bytes" -f $setupItem.Length)
Write-Host ("Version:    {0}" -f $version)
Write-Host ("Checksum:   {0}" -f $sumPath)
Write-Host ""
Write-Host "Smoke test next: run '$setupExe' to install, then open a new terminal and try 'transcribe --version'." -ForegroundColor Cyan
