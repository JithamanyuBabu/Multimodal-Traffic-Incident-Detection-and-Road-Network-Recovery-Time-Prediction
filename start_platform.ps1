# ============================================================================
# Multimodal Traffic Incident Detection & Network Recovery Platform
# Windows PowerShell One-Command Startup Script (start_platform.ps1)
# Usage: .\start_platform.ps1
# ============================================================================

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

Write-Host "==========================================================================" -ForegroundColor Cyan
Write-Host " Multimodal Traffic Incident Detection & Recovery Prediction Platform" -ForegroundColor Cyan
Write-Host "==========================================================================" -ForegroundColor Cyan

# 1. Locate Python executable (prefer local .venv if present)
$PythonExe = "python"
if (Test-Path "$ScriptDir\.venv\Scripts\python.exe") {
    $PythonExe = "$ScriptDir\.venv\Scripts\python.exe"
    Write-Host "[INFO] Using virtual environment Python: $PythonExe" -ForegroundColor Green
}

# 2. Verify required dependencies (install from requirements.txt if missing)
& $PythonExe "$ScriptDir\run_platform.py" --check-only
if ($LASTEXITCODE -ne 0) {
    Write-Host "[SETUP] Installing required Python packages from requirements.txt..." -ForegroundColor Yellow
    & $PythonExe -m pip install -r "$ScriptDir\requirements.txt"
}

# 3. Launch platform (seeds DB if needed, polls /api/health, opens browser at http://127.0.0.1:8000)
& $PythonExe "$ScriptDir\run_platform.py" @args
