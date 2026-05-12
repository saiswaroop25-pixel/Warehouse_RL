@echo off
cd /d "%~dp0"
if not exist "venv\Scripts\activate.bat" ( echo Run setup_env.bat first. & pause & exit /b 1 )
call venv\Scripts\activate.bat
set PYTHONUTF8=1
echo Starting evaluation for the latest run with live window ...
python evaluate.py --model_dir models --render --episodes 10 %*
pause
