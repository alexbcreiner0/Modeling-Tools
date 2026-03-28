$ErrorActionPreference = "Stop"

$AppRoot = $PSScriptRoot
$PythonExe = Join-Path $AppRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    Write-Error "Python environment not found at $PythonExe. The install step may have failed."
    exit 1
}

$env:PYTHONPATH = (Join-Path $AppRoot "src") + $(if ($env:PYTHONPATH) { ";$env:PYTHONPATH" } else { "" })

& $PythonExe -m modeling_tools @args