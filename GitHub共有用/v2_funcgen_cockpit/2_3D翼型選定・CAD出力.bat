@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" cockpit_selection_3d_cli.py
) else (
  python cockpit_selection_3d_cli.py
)
pause
