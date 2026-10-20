@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" grid_expander_3d_cli.py --run
) else (
  python grid_expander_3d_cli.py --run
)
pause
