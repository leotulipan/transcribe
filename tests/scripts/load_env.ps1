param (
    [Parameter(Mandatory=$true, Position=0)][string]$EnvFile,
    [Parameter(Mandatory=$false, Position=1)][string]$TargetKey
)

if (-Not (Test-Path -Path $EnvFile)) { return }

foreach ($line in Get-Content -Path $EnvFile) {
    if ([string]::IsNullOrWhiteSpace($line) -or $line.Trim().StartsWith("#")) { continue }

    $parts = $line -split '=', 2
    if ($parts.Length -ne 2) { continue }

    $key = $parts[0].Trim()
    $value = $parts[1].Trim().Trim('"', "'")

    if ([string]::IsNullOrEmpty($TargetKey) -or $key -eq $TargetKey) {
        Set-Item -Path "Env:$key" -Value $value
        if (-Not [string]::IsNullOrEmpty($TargetKey)) { break }
    }
}