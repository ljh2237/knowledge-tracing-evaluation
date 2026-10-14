@echo off
setlocal
cd /d "%~dp0.."
if not exist .venv\Scripts\python.exe call scripts\setup.bat
if errorlevel 1 exit /b 1
.venv\Scripts\python main.py --overwrite
if errorlevel 1 exit /b 1
pause
