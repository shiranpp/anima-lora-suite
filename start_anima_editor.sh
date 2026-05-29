#!/usr/bin/env bash
# --- Anima LoRA Editor - launch ------------------------------------------
# Runs the editor under the local ./venv. If you want Live Preview, install
# a CUDA build of torch into the venv yourself
# (https://pytorch.org/get-started/locally/) and then run ./setup_preview.sh
# for the rest of the generation extras. Otherwise preview reports
# "GPU required" and the editor continues to work fine without it.

set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PY="$HERE/venv/bin/python"

if [ ! -x "$LOCAL_PY" ]; then
    echo "[!] No local venv found at $LOCAL_PY"
    echo "    Run ./setup_env.sh first to create it. For Live Preview, also"
    echo "    install CUDA torch yourself + run ./setup_preview.sh."
    exit 1
fi

echo "[+] Python: $LOCAL_PY"
echo

exec "$LOCAL_PY" "$HERE/app.py" "$@"
