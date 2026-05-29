@echo off
REM --- Anima LoRA Editor - Windows setup --------------------------------
REM Creates a venv, installs requirements, leaves you ready to run app.py.

setlocal

echo.
echo  Anima LoRA Editor - Windows setup
echo  ----------------------------------
echo.

REM Check for python
where python >nul 2>nul
if errorlevel 1 (
    echo [!] Python not found on PATH. Install Python 3.10+ and retry.
    goto :fail
)

REM Create venv if missing
if not exist "venv\" (
    echo [+] Creating virtual environment in .\venv
    python -m venv venv
    if errorlevel 1 (
        echo [!] venv creation failed.
        goto :fail
    )
) else (
    echo [.] venv already exists, skipping creation
)

echo [+] Activating venv and upgrading pip
call venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet

echo [+] Installing requirements
pip install -r requirements.txt
if errorlevel 1 (
    echo [!] pip install failed.
    goto :fail
)

echo.
echo  Done. To launch:
echo      venv\Scripts\activate
echo      python app.py
echo.
echo  Or just run:  start_anima_editor.bat
echo.
echo  For Live Preview ^(real Anima image generation^) you also need a CUDA
echo  GPU. The easiest path is to run setup_preview.bat - it can install a
echo  CUDA build of torch into this venv for you, then add the extras.
echo.
pause
endlocal
exit /b 0

:fail
echo.
echo  Setup did not complete. See the message above.
pause
endlocal
exit /b 1
