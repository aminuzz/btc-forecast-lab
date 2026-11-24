@echo off
rem One-shot BTC forecast 15 minutes ahead.
cd /d "%~dp0"
set KRONOS_DIR=%~dp0Kronos-repo
set HF_HOME=%~dp0hf_cache
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
.\.venv\Scripts\python.exe predict_now.py
echo.
pause
