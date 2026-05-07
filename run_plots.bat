@echo off
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Run setup_env.bat first.
    pause & exit /b 1
)
call venv\Scripts\activate.bat
set PYTHONUTF8=1
echo.
echo ============================================================
echo   TA-RWARE Pro v2  -  Plot Training Results
echo ============================================================
echo.
python plot_results.py
echo.
echo Plots saved inside the latest run folder under logs\runs\<run_name>\plots\
echo Open the PNG files there to view your training graphs.
echo.
pause
