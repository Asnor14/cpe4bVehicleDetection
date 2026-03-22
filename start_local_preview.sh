#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$PROJECT_DIR/.runtime-venv/bin/python"
LOG_DIR="/home/raspi/.local/state/vehicle-detection"
LOG_FILE="$LOG_DIR/autostart.log"

mkdir -p "$LOG_DIR"

cd "$PROJECT_DIR"

# Give the desktop session a moment to finish initializing before OpenCV creates a window.
sleep 8

exec "$PYTHON_BIN" "$PROJECT_DIR/main.py" >>"$LOG_FILE" 2>&1
