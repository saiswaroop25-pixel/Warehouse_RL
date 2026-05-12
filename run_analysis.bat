@echo off
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" ( echo Run setup_env.bat first. & pause & exit /b 1 )
set PYTHONUTF8=1
echo Running deterministic DQN vs baseline comparison...
venv\Scripts\python.exe evaluate.py --model_dir models\runs\train_20260503_110644 --config configs\config.yaml --compare --seeds 1234,1235,1236
pause
