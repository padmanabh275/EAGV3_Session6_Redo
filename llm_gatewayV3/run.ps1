$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if (Get-Command py -ErrorAction SilentlyContinue) {
    $UsePyLauncher = $true
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $UsePyLauncher = $false
} else {
    throw "Neither 'py' nor 'python' was found in PATH."
}

if (-not (Test-Path ".venv")) {
    if ($UsePyLauncher) {
        py -3 -m venv .venv
    } else {
        python -m venv .venv
    }
}

& ".\.venv\Scripts\python.exe" -m pip install -q -r requirements.txt
& ".\.venv\Scripts\python.exe" main.py
