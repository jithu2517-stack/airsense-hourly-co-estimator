@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    py -3.11 -m venv .venv
    if errorlevel 1 goto :error
)

".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error

".venv\Scripts\python.exe" -m streamlit run app.py
exit /b %errorlevel%

:error
echo.
echo The demo could not start. Check that Python 3.11 is installed and that this computer can download the packages.
pause
exit /b 1
