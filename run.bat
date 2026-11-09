@echo off
chcp 65001 >nul 2>&1
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0"

echo ============================================
echo   ZhiXun Engine - one-click demo
echo ============================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] "python" not found in PATH.
    echo Install Anaconda or run with full path:
    echo     D:\anaconda\python.exe scripts\run_pipeline.py
    pause
    exit /b 1
)

echo [1/3] generating drift stream ...
python scripts\make_data.py
if errorlevel 1 goto fail

echo.
echo [2/3] running closed loop ...
python scripts\run_pipeline.py
if errorlevel 1 goto fail

echo.
echo [3/3] done. results are in out\
echo       curve.png     - baseline vs our system
echo       metrics.json  - per-batch scores
echo       decisions.jsonl - every decision record
echo.
if exist "out\curve.png" start "" "out\curve.png"
pause
exit /b 0

:fail
echo.
echo [FAILED] see the error above. Take a screenshot.
pause
exit /b 1
