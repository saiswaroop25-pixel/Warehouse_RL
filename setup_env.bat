@echo off
REM ============================================================
REM  TA-RWARE Pro v2  -  Windows Setup  (run ONCE)
REM ============================================================
echo.
echo ============================================================
echo   TA-RWARE Pro v2  -  Environment Setup
echo ============================================================
echo.
echo [1/5] Creating virtual environment ...
python -m venv venv
if errorlevel 1 ( echo ERROR: Python not found on PATH. & pause & exit /b 1 )
echo [2/5] Activating ...
call venv\Scripts\activate.bat
echo [3/5] Upgrading pip ...
python -m pip install --upgrade pip --quiet
echo [4/5] Installing packages ...
pip install -r requirements.txt --quiet
if errorlevel 1 ( echo ERROR: Install failed. & pause & exit /b 1 )
echo [5/5] Verifying ...
python -c "import torch,gymnasium,pygame,numpy,yaml,tqdm; print('All OK')"
if errorlevel 1 ( echo ERROR: Verification failed. & pause & exit /b 1 )
echo.
echo ============================================================
echo  Done!  TRAIN: run_train.bat   EVAL: run_evaluate.bat
echo ============================================================
pause
