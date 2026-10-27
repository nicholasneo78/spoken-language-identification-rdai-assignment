#!/usr/bin/env bash
# Creates ./venv and installs all requirements.
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv venv
./venv/bin/python -m pip install --upgrade pip
./venv/bin/python -m pip install -r requirements.txt
echo "Done. Activate with: source venv/bin/activate"
