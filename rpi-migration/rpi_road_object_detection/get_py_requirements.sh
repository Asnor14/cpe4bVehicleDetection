#!/bin/bash
set -euo pipefail

# Modern dependency install for webcam-only TFLite inference.
# Run inside a virtual environment for best results.

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "No virtualenv detected. Creating .venv in the repo..."
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

echo "Dependencies installed in: ${VIRTUAL_ENV}"
