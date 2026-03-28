$ErrorActionPreference = "Stop"

$InstallRoot = $PSScriptRoot
echo $InstallRoot
$LogFile = Join-Path $InstallRoot "install-log.txt"

Start-Transcript -Path $LogFile -Force | Out-Null

try {
    function Get-PythonCommand {
        if (Get-Command python -ErrorAction SilentlyContinue) {
            return @{
                Exe  = "python"
                Args = @()
            }
        }
        elseif (Get-Command py -ErrorAction SilentlyContinue) {
            return @{
                Exe  = "py"
                Args = @("-3")
            }
        }
        else {
            throw "Python 3.10+ is required, but no Python executable was found in PATH."
        }
    }

    $PythonCmd = Get-PythonCommand
    $PythonExe = $PythonCmd.Exe
    $PythonArgs = $PythonCmd.Args

    Write-Host "Python command: $PythonExe $($PythonArgs -join ' ')"

    $Version = & $PythonExe @PythonArgs -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"

    if (-not $Version) {
        throw "Could not determine Python version."
    }

    Write-Host "Detected Python version: $Version"

    $parts = $Version.Trim().Split(".")
    $major = [int]$parts[0]
    $minor = [int]$parts[1]

    if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
        throw "Python 3.10 or newer is required. Detected: $Version"
    }

    $VenvDir = Join-Path $InstallRoot ".venv"
    $VenvPython = Join-Path $VenvDir "Scripts\python.exe"

    if (Test-Path $VenvDir) {
        Write-Host "Removing existing virtual environment..."
        Remove-Item -Recurse -Force $VenvDir
    }

    Write-Host "Creating virtual environment at $VenvDir"
    & $PythonExe @PythonArgs -m venv $VenvDir

    if (-not (Test-Path $VenvPython)) {
        throw "Virtual environment creation failed. Missing: $VenvPython"
    }

    Write-Host "Upgrading pip..."
    & $VenvPython -m pip install --upgrade pip

    Write-Host "Installing package from $InstallRoot"
    & $VenvPython -m pip install $InstallRoot

    Write-Host "Installation complete."
}
catch {
    Write-Host ""
    Write-Host "INSTALL FAILED:"
    Write-Host $_
    Write-Host ""
    Write-Host "See log file: $LogFile"
    throw
}
finally {
    Stop-Transcript | Out-Null
}