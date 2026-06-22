param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("cpu", "gpu")]
    [string]$Profile
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$profilesDir = Join-Path $repoRoot ".devcontainer\profiles"
$source = Join-Path $profilesDir "devcontainer.$Profile.json"
$target = Join-Path $repoRoot ".devcontainer\devcontainer.json"

if (-not (Test-Path $source)) {
    throw "Profile file not found: $source"
}

Copy-Item -Path $source -Destination $target -Force

Write-Host "Switched active Dev Container profile to '$Profile'." -ForegroundColor Green
Write-Host "Active config: $target"
Write-Host "Next: run 'Dev Containers: Rebuild and Reopen in Container'."
