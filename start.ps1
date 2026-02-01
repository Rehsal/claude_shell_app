# PowerShell Start Script for Claude Shell App
# Run this from the claude_shell_app directory

cd $PSScriptRoot

# Kill any existing server on port 8000
$existing = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
foreach ($pid in $existing) {
    Write-Host "Killing old server process $pid"
    Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
}

.\venv\Scripts\Activate.ps1
python -m uvicorn app:app --host 0.0.0.0 --port 8000
