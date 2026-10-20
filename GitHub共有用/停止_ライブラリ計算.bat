@echo off
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0v2_funcgen_cockpit\stop_library.ps1"
pause
