@echo off
if not exist "venv\Scripts\activate.bat" ( echo Run setup_env.bat first. & pause & exit /b 1 )
call venv\Scripts\activate.bat
set PYTHONUTF8=1
echo.
echo ============================================================
echo   TA-RWARE Pro v2  -  Training
echo ============================================================
echo.
echo   1  =  Start FRESH training  (creates a new run folder)
echo   2  =  RESUME from last checkpoint
echo.
set /p ch="Choice (1 or 2): "
if "%ch%"=="1" (
    echo Starting fresh training in a new timestamped run ...
    python train.py
)
if "%ch%"=="2" (
    echo Resuming from checkpoint ...
    python train.py --resume
)
if not "%ch%"=="1" if not "%ch%"=="2" (
    echo Invalid choice.
)
pause
