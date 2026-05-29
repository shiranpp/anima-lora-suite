@echo off
REM --- Anima LoRA Editor - launch ------------------------------------------
REM Runs the editor under the local .\venv. If you want Live Preview, install
REM a CUDA build of torch into the venv yourself
REM (https://pytorch.org/get-started/locally/) and then run setup_preview.bat
REM for the rest of the generation extras. Otherwise preview reports
REM "GPU required" and the editor continues to work fine without it.

setlocal

set "LOCAL_PY=%~dp0venv\Scripts\python.exe"

if not exist "%LOCAL_PY%" (
    echo [!] No local venv found at %LOCAL_PY%
    echo     Run setup_env.bat first to create it. For Live Preview, also
    echo     install CUDA torch yourself + run setup_preview.bat.
    pause
    exit /b 1
)

echo [+] Python: %LOCAL_PY%
echo.

"%LOCAL_PY%" "%~dp0app.py" %*
set "RC=%errorlevel%"
echo.
if not "%RC%"=="0" echo [!] Editor exited with code %RC%.
pause
exit /b %RC%
