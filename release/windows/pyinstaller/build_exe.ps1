param(
    [string]$AppName
)

if (-not $AppName) {
    Write-Host "Usage: .\build_exe.ps1 APP_NAME"
    exit 1
}

Set-Location $PSScriptRoot

pyinstaller `
  -n $AppName `
  --clean `
  --noconfirm `
  --additional-hooks-dir=. `
  --collect-data modeling_tools `
  --collect-data scienceplots `
  --hidden-import modeling_tools.tools.log_formatter `
  --paths ..\..\..\src `
  .\main.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "PyInstaller build failed."
    exit 1
}