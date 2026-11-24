@echo off
rem Serves the dashboard at http://localhost:8765/ reading .\follow_ledger.json
cd /d "%~dp0"
start "" http://localhost:8765/dashboard.html
.\.venv\Scripts\python.exe -m http.server 8765
