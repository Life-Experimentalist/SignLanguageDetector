param(
    [string]$PythonVersion = "3.12",
    [switch]$Recreate,
    [switch]$InstallDev
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv is not installed. Install it from https://docs.astral.sh/uv/"
    exit 1
}

if ($Recreate -and (Test-Path ".venv")) {
    Write-Host "Removing existing .venv..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force ".venv"
}

Write-Host "Creating .venv with Python $PythonVersion..." -ForegroundColor Cyan
uv venv --python $PythonVersion .venv

if (Test-Path "pyproject.toml") {
    Write-Host "Syncing dependencies from pyproject.toml..." -ForegroundColor Cyan
    if ($InstallDev) {
        uv sync --python .venv\Scripts\python.exe --group dev
    }
    else {
        uv sync --python .venv\Scripts\python.exe
    }
}
elseif (Test-Path "requirements.txt") {
    Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Cyan
    uv pip install --python .venv\Scripts\python.exe -r requirements.txt
}
else {
    Write-Error "No pyproject.toml or requirements.txt found."
    exit 1
}

Write-Host "Setup completed." -ForegroundColor Green
Write-Host "Activate with: .\.venv\Scripts\Activate.ps1"