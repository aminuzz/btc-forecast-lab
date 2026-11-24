@echo off
rem Starts the interactive Kronos bot at http://localhost:8766/
cd /d "%~dp0"
set KRONOS_DIR=%~dp0Kronos-repo
set HF_HOME=%~dp0hf_cache
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
start "" http://localhost:8766/
.\.venv\Scripts\python.exe app.py
