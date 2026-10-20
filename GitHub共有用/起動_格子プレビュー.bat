@echo off
chcp 65001 >nul
cd /d "%~dp0v2_funcgen_cockpit"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" grid_expander_3d_cli.py --dry-run
) else (
  python grid_expander_3d_cli.py --dry-run
)
pause
