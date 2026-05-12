@echo off
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo Virtual environment not found. Run setup_env.bat first.
    pause
    exit /b 1
)

if not exist "venv\Scripts\streamlit.exe" (
    echo Streamlit is not installed in the virtual environment.
    echo Installing project requirements...
    venv\Scripts\python.exe -m pip install -r requirements.txt
)

if not exist "models\agent0_best.pt" (
    echo Trained model checkpoints were not found in the models folder.
    echo Run training first, or copy the trained models folder into this project.
    pause
    exit /b 1
)

echo Opening TA-RWARE presentation dashboard...
echo Use the Run tab for live movement and Compare policies for DQN vs baselines.
venv\Scripts\streamlit.exe run streamlit_app.py
