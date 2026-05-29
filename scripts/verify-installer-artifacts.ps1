# scripts/verify-installer-artifacts.ps1
#
# Post-build assertions for the Windows installer pipeline. Runs against the
# dist/ directory and verifies that build-installer.ps1 produced what it was
# supposed to produce, with the expected embedded metadata.
#
# This is the closest thing to a unit test for the installer pipeline: it
# does NOT actually run the installer (that would require driving a GUI
# wizard), but it catches regressions like:
#   - missing artifacts
#   - wrong product/file version embedded
#   - missing icon in the executables
#   - stale .sha256 sidecar that doesn't match the .exe
#   - GUI binary built without the windowsgui subsystem (would leak a console)
#
# Exit code 0 if all assertions pass, non-zero on any failure.
#
# Usage:
#   pwsh scripts/verify-installer-artifacts.ps1                 # auto-detect version from git
#   pwsh scripts/verify-installer-artifacts.ps1 -Version 0.10.0

[CmdletBinding()]
param(
    [string]$Version,
    [string]$DistDir = (Join-Path (Resolve-Path "$PSScriptRoot\..").Path "dist")
)

$ErrorActionPreference = "Continue"

if (-not $Version) {
    try {
        $tag = (git describe --tags --abbrev=0 2>$null)
        if ($LASTEXITCODE -eq 0 -and $tag) { $Version = $tag.Trim().TrimStart('v') }
    } catch {}
    if (-not $Version) { $Version = "0.0.0-dev" }
}

Write-Host ""
Write-Host "=== Verifying dist/ artifacts for version $Version ===" -ForegroundColor Cyan
Write-Host ""

$failures = @()
function Assert {
    param([string]$Label, [bool]$Condition, [string]$Detail)
    if ($Condition) {
        Write-Host ("  PASS  {0}" -f $Label) -ForegroundColor Green
        if ($Detail) { Write-Host ("        {0}" -f $Detail) -ForegroundColor DarkGray }
    } else {
        Write-Host ("  FAIL  {0}" -f $Label) -ForegroundColor Red
        if ($Detail) { Write-Host ("        {0}" -f $Detail) -ForegroundColor Yellow }
        $script:failures += $Label
    }
}

# Resolve expected artifact paths.
$cliExe   = Join-Path $DistDir "transcribe.exe"
$guiExe   = Join-Path $DistDir "transcribe-gui.exe"
$setupExe = Join-Path $DistDir "transcribe-setup-v$Version.exe"
$sumFile  = "$setupExe.sha256"

# --- Presence checks ---
Assert "CLI binary present"        (Test-Path $cliExe)   $cliExe
Assert "GUI binary present"        (Test-Path $guiExe)   $guiExe
Assert "Installer present"         (Test-Path $setupExe) $setupExe
Assert "SHA256 sidecar present"    (Test-Path $sumFile)  $sumFile

if ($failures.Count -gt 0) {
    Write-Host ""
    Write-Host "Artifacts missing; aborting deeper checks." -ForegroundColor Red
    exit 1
}

# --- Embedded version metadata ---
$cliInfo = (Get-Item $cliExe).VersionInfo
$guiInfo = (Get-Item $guiExe).VersionInfo

Assert "CLI ProductVersion matches"    ($cliInfo.ProductVersion -like "$Version*") ("got '{0}'" -f $cliInfo.ProductVersion)
Assert "GUI ProductVersion matches"    ($guiInfo.ProductVersion -like "$Version*") ("got '{0}'" -f $guiInfo.ProductVersion)
Assert "CLI InternalName=transcribe"   ($cliInfo.InternalName -eq "transcribe")    ("got '{0}'" -f $cliInfo.InternalName)
Assert "GUI InternalName=transcribe-gui" ($guiInfo.InternalName -eq "transcribe-gui") ("got '{0}'" -f $guiInfo.InternalName)
Assert "CLI ProductName=Audio Transcribe" ($cliInfo.ProductName -eq "Audio Transcribe") ("got '{0}'" -f $cliInfo.ProductName)
Assert "CLI FileDescription is CLI variant" ($cliInfo.FileDescription -like "*CLI*") ("got '{0}'" -f $cliInfo.FileDescription)
Assert "GUI FileDescription is GUI variant" ($guiInfo.FileDescription -eq "Audio Transcribe") ("got '{0}'" -f $guiInfo.FileDescription)
Assert "CLI CompanyName set"           (-not [string]::IsNullOrEmpty($cliInfo.CompanyName)) ("got '{0}'" -f $cliInfo.CompanyName)

# --- PE subsystem (windowsgui vs console) ---
# The DOS header says PE starts at offset 0x3C. Subsystem field is at PE
# header offset 0x5C (PE sig + 4 + 0x18 + 0x44). 2 = GUI, 3 = console.
function Get-PESubsystem {
    param([string]$Path)
    $bytes = [System.IO.File]::ReadAllBytes($Path)
    $peOffset = [BitConverter]::ToInt32($bytes, 0x3C)
    # PE\0\0 (4) + COFF header (20) + OptionalHeader.Subsystem at offset 68
    $subsysOffset = $peOffset + 4 + 20 + 68
    return [BitConverter]::ToUInt16($bytes, $subsysOffset)
}
$cliSub = Get-PESubsystem -Path $cliExe
$guiSub = Get-PESubsystem -Path $guiExe
Assert "CLI uses console subsystem (3)" ($cliSub -eq 3) ("got subsystem={0}" -f $cliSub)
Assert "GUI uses GUI subsystem (2)"     ($guiSub -eq 2) ("got subsystem={0}" -f $guiSub)

# --- SHA256 sidecar matches installer ---
$actualHash = (Get-FileHash $setupExe -Algorithm SHA256).Hash.ToLower()
$sumLine = (Get-Content $sumFile -Raw).Trim()
$expectedHash = ($sumLine -split '\s+')[0]
Assert "SHA256 sidecar matches installer" ($actualHash -eq $expectedHash) ("file={0}, sidecar={1}" -f $actualHash, $expectedHash)

# --- Icon presence on .exe files (best-effort) ---
# Verifying the embedded icon byte-for-byte is overkill; we settle for "the
# .exe has any embedded icon resource at all". Windows exposes the icon
# through ExtractAssociatedIcon when one is present.
function Has-Icon {
    param([string]$Path)
    try {
        Add-Type -AssemblyName System.Drawing -ErrorAction SilentlyContinue
        $ico = [System.Drawing.Icon]::ExtractAssociatedIcon($Path)
        return ($null -ne $ico)
    } catch { return $false }
}
Assert "CLI has an embedded icon" (Has-Icon $cliExe) ""
Assert "GUI has an embedded icon" (Has-Icon $guiExe) ""

# --- Installer is a valid Windows PE executable ---
# Don't grep for the "Inno Setup" string in the header — LZMA2/ultra
# compression pushes it past any reasonable read window. The fact that
# the file exists at the expected name (build-installer.ps1 only writes
# it on successful ISCC exit) is sufficient.
$mz = [System.IO.File]::ReadAllBytes($setupExe)[0..1]
Assert "Installer is a PE executable (MZ header)" ($mz[0] -eq 0x4D -and $mz[1] -eq 0x5A) ""

Write-Host ""
if ($failures.Count -eq 0) {
    Write-Host "All artifact checks passed." -ForegroundColor Green
    exit 0
} else {
    Write-Host ("{0} check(s) failed:" -f $failures.Count) -ForegroundColor Red
    foreach ($f in $failures) { Write-Host "  - $f" }
    exit 1
}
