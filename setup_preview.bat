@echo off
REM --- Anima LoRA Editor - REAL preview extras (Windows) ----------------
REM Installs the *non-torch* generation deps for the Live Preview panel,
REM and can install a CUDA build of torch into this venv for you.
REM
REM IMPORTANT: torch must live INSIDE this project's .\venv. Running a plain
REM "pip install ... torch" from a normal prompt installs it into your system
REM Python instead, which does NOT help this venv. This script always targets
REM the venv, so let it do the install (or use venv\Scripts\python -m pip ...).

setlocal enabledelayedexpansion

echo.
echo  Anima LoRA Editor - preview (real generation) setup
echo  ----------------------------------------------------
echo.

if not exist "venv\Scripts\activate.bat" (
    echo [!] venv not found. Run setup_env.bat first.
    goto :fail
)

call venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet

REM Probe torch inside the venv. Exit codes:
REM   0 = torch present and CUDA available
REM   2 = torch present but CUDA NOT available (e.g. a +cpu build)
REM   1 = torch not installed (import failed)
call :probe_torch
if "%TORCH_RC%"=="0" goto :torch_ok

REM Grab the installed version (if any) for a clearer message.
set "TORCH_VER="
for /f "delims=" %%V in ('python -c "import torch; print(torch.__version__)" 2^>nul') do set "TORCH_VER=%%V"

echo.
if "%TORCH_RC%"=="2" (
    echo [!] torch is installed in the venv ^(%TORCH_VER%^) but cannot use CUDA.
    echo     This is almost always a CPU-only build ^(version ends in +cpu^).
) else (
    echo [!] torch is not installed in this venv.
)
echo     Anima preview needs a CUDA GPU and a matching CUDA build of torch
echo     installed INTO THIS VENV ^(.\venv^), not your system Python.
echo.

REM Offer to install the CUDA wheel into the venv automatically.
set "CUDA_TAG=cu128"
echo     Pick the CUDA wheel tag that matches your driver
echo     ^(see https://pytorch.org/get-started/locally/ - e.g. cu121, cu124, cu128^).
set /p "CUDA_TAG=    CUDA tag to install [%CUDA_TAG%], or N to skip: "

if /i "%CUDA_TAG%"=="N" (
    echo.
    echo     Skipped. Install it yourself into the venv with:
    echo       venv\Scripts\python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch
    echo     then re-run this script.
    goto :fail
)
if not defined CUDA_TAG (
    echo [!] No CUDA tag entered. Re-run and choose a tag ^(e.g. cu128^) or N to skip.
    goto :fail
)

echo.
echo [+] Reinstalling torch from https://download.pytorch.org/whl/%CUDA_TAG% into the venv
REM --force-reinstall so an existing CPU build is replaced by the CUDA wheel.
python -m pip install --index-url https://download.pytorch.org/whl/%CUDA_TAG% --force-reinstall torch
if errorlevel 1 (
    echo [!] torch install failed. Check that the CUDA tag "%CUDA_TAG%" is valid.
    goto :fail
)

echo [+] Re-checking torch / CUDA
call :probe_torch
if "%TORCH_RC%"=="0" goto :torch_ok

echo.
echo [!] torch still cannot use CUDA after installing the "%CUDA_TAG%" wheel.
echo     Likely causes: wrong CUDA tag for your driver, or no CUDA GPU/driver.
echo     Check your driver and the right tag at:
echo       https://pytorch.org/get-started/locally/
echo     then re-run this script.
goto :fail

:torch_ok
for /f "delims=" %%V in ('python -c "import torch; print(torch.__version__)"') do set "TORCH_VER=%%V"
echo [+] torch %TORCH_VER% ^(CUDA ok^) - proceeding

echo [+] Installing generation extras
pip install -r requirements-preview.txt
if errorlevel 1 (
    echo [!] pip install failed.
    goto :fail
)

echo.
echo  Done. Open the Live Preview panel, expand "Model paths", and point it
echo  at your Anima DiT / VAE / Qwen3 files to generate real samples.
echo.
goto :done

:fail
echo.
echo  Setup did not complete. See the message above.
pause
endlocal
exit /b 1

:done
pause
endlocal
exit /b 0

REM -- helper: set TORCH_RC to 0 (CUDA ok) / 2 (no CUDA) / 1 (not installed) --
:probe_torch
python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 2)" 2>nul
set "TORCH_RC=%errorlevel%"
goto :eof
