#!/usr/bin/env bash
# ─── Anima LoRA Editor — REAL preview extras (Linux/macOS) ─────────────
# Installs the *non-torch* generation deps for the Live Preview panel.
#
# torch + CUDA is the user's responsibility — install the CUDA wheel from
# https://pytorch.org/get-started/locally/ into this venv *before* running
# this script. (Anima generation needs a CUDA GPU; there is no CPU path.)

set -e

echo
echo "  Anima LoRA Editor — preview (real generation) setup"
echo "  ----------------------------------------------------"
echo

if [ ! -d "venv" ]; then
    echo "[!] venv not found. Run ./setup_env.sh first."
    exit 1
fi

# shellcheck disable=SC1091
source venv/bin/activate
python -m pip install --upgrade pip --quiet

# Bail early with a clear pointer if torch isn't installed or isn't CUDA-capable.
# Anima preview needs a real CUDA GPU; we don't try to install torch for the
# user because the right wheel depends on their CUDA toolkit version.
if ! python - <<'PY' ; then
import sys
try:
    import torch
except ImportError:
    print("[!] torch is not installed in this venv.")
    print("    Install a CUDA build matching your GPU from:")
    print("    https://pytorch.org/get-started/locally/")
    print("    e.g.  pip install --index-url https://download.pytorch.org/whl/cu128 torch")
    sys.exit(1)
if not torch.cuda.is_available():
    print(f"[!] torch {torch.__version__} is installed but CUDA is not available.")
    print("    Anima preview needs a CUDA GPU. Reinstall torch from the CUDA")
    print("    wheel index that matches your driver:")
    print("    https://pytorch.org/get-started/locally/")
    sys.exit(1)
print(f"[+] torch {torch.__version__} (CUDA ok) — proceeding")
PY
    exit 1
fi

echo "[+] Installing generation extras"
pip install -r requirements-preview.txt

echo
echo "  Done. Open the Live Preview panel, expand \"Model paths\", and point"
echo "  it at your Anima DiT / VAE / Qwen3 files to generate real samples."
echo
