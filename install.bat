@echo off
REM Installs the Python packages needed by shaft_optimizer.py
cd /d "%~dp0"
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo Install failed. Make sure Python 3.10+ from python.org is installed
  echo with "Add python.exe to PATH" and "tcl/tk and IDLE" checked.
) else (
  echo.
  echo Done. Double-click run.bat to start the optimizer.
)
pause
