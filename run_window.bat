@echo off
rem Paper-trade the next N 15-minute KXBTC15M windows.
rem Usage: run_window.bat [windows] [decide-after-seconds]
cd /d "%~dp0"
set WINDOWS=%1
if "%WINDOWS%"=="" set WINDOWS=4
set OFFSET=%2
set OFFSET_ARG=
if not "%OFFSET%"=="" set OFFSET_ARG=--decide-after %OFFSET%
set KRONOS_DIR=%~dp0Kronos-repo
set HF_HOME=%~dp0hf_cache
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
.\.venv\Scripts\python.exe fifteen.py --windows %WINDOWS% %OFFSET_ARG%
