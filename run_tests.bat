@echo off
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" ( echo Run setup_env.bat first. & pause & exit /b 1 )
set PYTHONUTF8=1
venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
pause
