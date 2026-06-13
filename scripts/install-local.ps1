<#
.SYNOPSIS
    Install freshly built transcribe binaries into the local install locations
    and verify them. Run after scripts/build.ps1.

.DESCRIPTION
    Stops any running transcribe / transcribe-gui processes (so the .exe files
    aren't locked), copies transcribe.exe and transcribe-gui.exe from the build
    output directory into each install location that exists, then prints the
    installed version and checks that the `merge` subcommand is present.

    Pairing build + install as a single script keeps the whole flow behind one
    allowlisted command, avoiding the permission prompts and PowerShell parser
    timeouts that ad-hoc inline build/copy/verify one-liners trigger.

.PARAMETER SourceDir
    Directory containing the built binaries. Defaults to "bin" (build.ps1's
    default output directory).

.EXAMPLE
    ./scripts/build.ps1 -Version v0.11.0
    ./scripts/install-local.ps1
#>
param(
    [string]$SourceDir = "bin"
)
$ErrorActionPreference = "Stop"

$cli = Join-Path $SourceDir "transcribe.exe"
$gui = Join-Path $SourceDir "transcribe-gui.exe"
foreach ($f in @($cli, $gui)) {
    if (-not (Test-Path $f)) {
        throw "Missing '$f' - run scripts/build.ps1 first."
    }
}

# Stop running instances so the binaries aren't locked during copy. The
# binaries are named transcribe / transcribe-gui, never this shell.
Get-Process transcribe, transcribe-gui -ErrorAction SilentlyContinue |
    Stop-Process -Force -ErrorAction SilentlyContinue

$targets = @(
    (Join-Path $env:USERPROFILE ".local\bin"),
    (Join-Path $env:LOCALAPPDATA "Programs\Transcribe")
)

$installed = 0
foreach ($t in $targets) {
    if (Test-Path $t) {
        Copy-Item $cli (Join-Path $t "transcribe.exe")     -Force
        Copy-Item $gui (Join-Path $t "transcribe-gui.exe") -Force
        Write-Host "installed -> $t"
        $installed++
    }
}
if ($installed -eq 0) {
    Write-Warning "No install directories found; nothing copied. Checked:`n  $($targets -join "`n  ")"
    return
}

Write-Host "`n=== verify ==="
& $cli -V
$help = (& $cli --help 2>&1 | Out-String)
if ($help -match "merge") {
    Write-Host "merge subcommand: present"
} else {
    Write-Warning "merge subcommand NOT found in --help"
}
