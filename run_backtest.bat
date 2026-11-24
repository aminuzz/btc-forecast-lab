@echo off
rem Run a backtest config. Usage: run_backtest.bat [n_events] [paths] [lookback] [model] [offset]
cd /d "%~dp0"
set N=%1
if "%N%"=="" set N=288
set PATHS=%2
if "%PATHS%"=="" set PATHS=32
set LB=%3
if "%LB%"=="" set LB=240
set MODEL=%4
if "%MODEL%"=="" set MODEL=small
set OFF=%5
if "%OFF%"=="" set OFF=60
set KRONOS_DIR=%~dp0Kronos-repo
set HF_HOME=%~dp0hf_cache
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
.\.venv\Scripts\python.exe backtest.py --n-events %N% --paths %PATHS% --lookback %LB% --model %MODEL% --offset %OFF%
