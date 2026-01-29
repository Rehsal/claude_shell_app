# PowerShell Start Script for Claude Shell App
# Run this from the claude_shell_app directory

cd $PSScriptRoot
.\venv\Scripts\Activate.ps1
python -m uvicorn app:app --host 0.0.0.0 --port 8000
