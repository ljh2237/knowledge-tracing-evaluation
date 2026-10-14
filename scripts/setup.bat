@echo off
setlocal
cd /d "%~dp0.."
py -3.12 -m venv .venv
if errorlevel 1 exit /b 1
.venv\Scripts\python -m pip install -r requirements-lock.txt
if errorlevel 1 exit /b 1
echo Ready. Run .venv\Scripts\python main.py
