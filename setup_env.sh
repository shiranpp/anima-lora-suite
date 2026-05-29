#!/usr/bin/env bash
# ─── Anima LoRA Editor — Linux/macOS setup ─────────────────────────────
# Creates a venv, installs requirements, leaves you ready to run app.py.

set -e

echo
echo "  Anima LoRA Editor — Linux/macOS setup"
echo "  --------------------------------------"
echo

if ! command -v python3 >/dev/null 2>&1; then
    echo "[!] python3 not found on PATH. Install Python 3.10+ and retry."
    exit 1
fi

PYTHON="$(command -v python3)"
echo "[+] Using $($PYTHON --version) at $PYTHON"

if [ ! -d "venv" ]; then
    echo "[+] Creating virtual environment in ./venv"
    "$PYTHON" -m venv venv
else
    echo "[.] venv already exists, skipping creation"
fi

# shellcheck disable=SC1091
source venv/bin/activate

echo "[+] Upgrading pip"
python -m pip install --upgrade pip --quiet

echo "[+] Installing requirements"
pip install -r requirements.txt

echo
echo "  Done. To launch:"
echo "      source venv/bin/activate"
echo "      python app.py"
echo
echo "  Or just run:  ./start_anima_editor.sh"
echo
echo "  For Live Preview (real Anima image generation) you also need a CUDA"
echo "  GPU. Install a CUDA build of torch into this venv yourself:"
echo "      https://pytorch.org/get-started/locally/"
echo "  then run ./setup_preview.sh for the rest of the generation extras."
echo
