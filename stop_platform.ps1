# ============================================================================
# Multimodal Traffic Incident Detection & Network Recovery Platform
# Windows PowerShell Shutdown Script (stop_platform.ps1)
# Cleanly stops any process listening on Port 8000.
# ============================================================================

param(
    [int]$Port = 8000
)

Write-Host "[STOP] Checking for active platform server on port $Port..." -ForegroundColor Yellow
$Connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($Connections) {
    $Pids = $Connections | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($ProcessId in $Pids) {
        if ($ProcessId -gt 0) {
            Write-Host "[STOP] Terminating process PID $ProcessId listening on port $Port..." -ForegroundColor Cyan
            Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
        }
    }
    Write-Host "[OK] Port $Port released cleanly." -ForegroundColor Green
} else {
    Write-Host "[OK] No active server listening on port $Port." -ForegroundColor Green
}
