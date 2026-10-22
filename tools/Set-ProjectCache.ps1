param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ProjectCommand
)

$TaskRepoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$TaskCacheRoot = [IO.Path]::GetFullPath((Join-Path $TaskRepoRoot '.cache'))
if (-not $TaskCacheRoot.StartsWith($TaskRepoRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Cache root must be inside this repository.'
}
if ((Test-Path -LiteralPath $TaskCacheRoot) -and
    ((Get-Item -LiteralPath $TaskCacheRoot -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
    throw 'Cache root must not redirect to another location.'
}
$TaskCacheVariables = @{
    TEMP = 'temp'
    TMP = 'temp'
    TMPDIR = 'temp'
    PIP_CACHE_DIR = 'pip'
    MPLCONFIGDIR = 'matplotlib'
    PYTHONPYCACHEPREFIX = 'python'
    XDG_CACHE_HOME = 'xdg'
    NUMBA_CACHE_DIR = 'numba'
    JOBLIB_TEMP_FOLDER = 'joblib'
}
foreach ($TaskEntry in $TaskCacheVariables.GetEnumerator()) {
    $TaskCachePath = Join-Path $TaskCacheRoot $TaskEntry.Value
    if ((Test-Path -LiteralPath $TaskCachePath) -and
        ((Get-Item -LiteralPath $TaskCachePath -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw 'Cache directories must not redirect to another location.'
    }
    New-Item -ItemType Directory -Path $TaskCachePath -Force | Out-Null
    [Environment]::SetEnvironmentVariable($TaskEntry.Key, $TaskCachePath, 'Process')
}
if ($ProjectCommand.Count -gt 0) {
    $TaskExecutable = $ProjectCommand[0]
    $TaskArguments = @($ProjectCommand | Select-Object -Skip 1)
    & $TaskExecutable @TaskArguments
    if ($LASTEXITCODE -ne 0) { throw "Project command exited with code $LASTEXITCODE" }
} else {
    Write-Output "Project caches and temporary files: $TaskCacheRoot"
}
