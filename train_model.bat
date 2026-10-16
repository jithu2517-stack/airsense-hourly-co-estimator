@echo off
setlocal
cd /d "%~dp0"

if "%~1"=="" (
    echo Usage: train_model.bat "C:\path\to\AirQualityUCI.csv"
    echo You can also provide an XLSX file or a ZIP containing the UCI Air Quality files.
    pause
    exit /b 2
)

if not exist ".venv\Scripts\python.exe" (
    py -3.11 -m venv .venv
    if errorlevel 1 goto :error
)

".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error

".venv\Scripts\python.exe" train_model.py --data "%~1" --output "%~dp0"
if errorlevel 1 goto :error
echo.
echo Model files and evaluation results have been rebuilt in this folder.
pause
exit /b 0

:error
echo.
echo Training did not finish. Check the dataset path and Python 3.11 installation.
pause
exit /b 1
