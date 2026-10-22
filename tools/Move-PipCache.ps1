param([Parameter(Mandatory = $true)][string]$Source)
$ErrorActionPreference = 'Stop'
$TaskRepoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$TaskExpectedSource = [IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA 'pip\Cache'))
$TaskSource = (Get-Item -LiteralPath $Source -Force).FullName
$TaskTarget = [IO.Path]::GetFullPath((Join-Path $TaskRepoRoot '.cache\pip'))
if ($TaskSource.TrimEnd('\') -ne $TaskExpectedSource.TrimEnd('\')) { throw 'Source must be the current user pip cache.' }
if (-not $TaskTarget.StartsWith($TaskRepoRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Destination must remain inside the repository.' }
if ((Get-Item -LiteralPath $TaskSource -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Source cache must not be a link.' }
if ((Get-ChildItem -LiteralPath $TaskSource -Recurse -Force | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count) { throw 'Source contains a link.' }
if (Test-Path -LiteralPath $TaskTarget) {
    if ((Get-ChildItem -LiteralPath $TaskTarget -Force).Count) { throw 'Destination cache must be empty for verified migration.' }
} else {
    New-Item -ItemType Directory -Path $TaskTarget -Force | Out-Null
}
if ((Get-Item -LiteralPath $TaskTarget -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Destination cache must not be a link.' }
$TaskReceiptFolder = Join-Path $TaskRepoRoot 'outputs\cache_storage_20261014'
New-Item -ItemType Directory -Path $TaskReceiptFolder -Force | Out-Null
$TaskReceiptPath = Join-Path $TaskReceiptFolder 'pip_cache_migration.json'
if (Test-Path -LiteralPath $TaskReceiptPath) { throw 'Existing migration receipt must be preserved.' }
$TaskFiles = @(Get-ChildItem -LiteralPath $TaskSource -Recurse -File -Force)
$TaskManifest = @{}
$TaskBytes = 0L
foreach ($TaskFile in $TaskFiles) {
    $TaskRelative = $TaskFile.FullName.Substring($TaskSource.TrimEnd('\').Length + 1)
    $TaskManifest[$TaskRelative] = @{ length = $TaskFile.Length; sha256 = (Get-FileHash -LiteralPath $TaskFile.FullName -Algorithm SHA256).Hash }
    $TaskBytes += $TaskFile.Length
}
Write-Output "Copying $($TaskFiles.Count) pip cache files ($TaskBytes bytes) to $TaskTarget"
Get-ChildItem -LiteralPath $TaskSource -Force | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination $TaskTarget -Recurse -Force
}
foreach ($TaskRelative in $TaskManifest.Keys) {
    $TaskCopiedFile = Join-Path $TaskTarget $TaskRelative
    if ((Get-Item -LiteralPath $TaskCopiedFile -Force).Length -ne $TaskManifest[$TaskRelative].length -or
        (Get-FileHash -LiteralPath $TaskCopiedFile -Algorithm SHA256).Hash -ne $TaskManifest[$TaskRelative].sha256) {
        throw 'Copy verification failed. Source cache retained.'
    }
}
$TaskCurrentFiles = @(Get-ChildItem -LiteralPath $TaskSource -Recurse -File -Force)
if ($TaskCurrentFiles.Count -ne $TaskFiles.Count) { throw 'Source cache changed during migration. Source retained.' }
foreach ($TaskFile in $TaskCurrentFiles) {
    $TaskRelative = $TaskFile.FullName.Substring($TaskSource.TrimEnd('\').Length + 1)
    if (-not $TaskManifest.ContainsKey($TaskRelative) -or
        (Get-FileHash -LiteralPath $TaskFile.FullName -Algorithm SHA256).Hash -ne $TaskManifest[$TaskRelative].sha256) {
        throw 'Source cache changed during migration. Source retained.'
    }
}
$TaskPreviousSetting = [Environment]::GetEnvironmentVariable('PIP_CACHE_DIR', 'User')
[Environment]::SetEnvironmentVariable('PIP_CACHE_DIR', $TaskTarget, 'User')
[Environment]::SetEnvironmentVariable('PIP_CACHE_DIR', $TaskTarget, 'Process')
# Delete only the verified pip-cache copies. Preserve the source directory itself.
if ((Get-Item -LiteralPath $TaskSource -Force).FullName -ne $TaskExpectedSource) { throw 'Source path changed; cleanup refused.' }
foreach ($TaskFile in $TaskCurrentFiles) {
    if (-not $TaskFile.FullName.StartsWith($TaskExpectedSource + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Cleanup path escaped the verified pip cache.' }
    Remove-Item -LiteralPath $TaskFile.FullName -Force
}
$TaskRecord = @{
    source = $TaskSource; destination = $TaskTarget; files = $TaskFiles.Count; bytes = $TaskBytes
    every_file_sha256_verified = $true; source_files_removed_after_verification = $true
    user_PIP_CACHE_DIR = $TaskTarget; previous_user_PIP_CACHE_DIR = $TaskPreviousSetting
    scientific_outputs_changed = $false; files_sha256 = $TaskManifest
}
$TaskRecord | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $TaskReceiptPath -Encoding UTF8
Write-Output "Moved and verified $($TaskFiles.Count) cache files. Future user pip cache: $TaskTarget"
