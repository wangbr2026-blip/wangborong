param(
    [ValidateSet('scan','reconstruct','figures','check')]
    [string]$Stage = 'check'
)
$ErrorActionPreference = 'Stop'
$taskPython = $env:NIR_PYTHON
if (-not $taskPython) {
    $runtime = Get-AppxPackage -Name 'OpenAI.CodexPrimaryRuntime*' |
        Sort-Object Version -Descending | Select-Object -First 1
    if ($runtime) {
        $taskPython = Join-Path $runtime.InstallLocation 'dependencies/python/python.exe'
    } else {
        $taskPython = (Get-Command python -ErrorAction Stop).Source
    }
}
$env:PYTHONIOENCODING = 'utf-8'
& $taskPython (Join-Path $PSScriptRoot 'stage2.py') $Stage
exit $LASTEXITCODE

