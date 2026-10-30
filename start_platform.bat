@echo off
REM ============================================================================
REM Multimodal Traffic Incident Detection & Network Recovery Platform
REM Windows Batch One-Command Launcher (start_platform.bat)
REM ============================================================================
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" run_platform.py %*
) else (
    python run_platform.py %*
)
