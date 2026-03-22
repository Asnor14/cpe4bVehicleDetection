#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.runtime-venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
AUTOSTART_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/autostart"
AUTOSTART_FILE="$AUTOSTART_DIR/vehicle-detection-local.desktop"
INSTALL_AUTOSTART=0

usage() {
    echo "Usage: ./install_on_rpi.sh [--autostart]"
    echo
    echo "Options:"
    echo "  --autostart   Install the desktop autostart entry for the current user."
    echo "  --help        Show this message."
}

while [ $# -gt 0 ]; do
    case "$1" in
        --autostart)
            INSTALL_AUTOSTART=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: $PYTHON_BIN was not found. Install Python 3 first." >&2
    exit 1
fi

echo "Project directory: $PROJECT_DIR"
echo "Using Python: $(command -v "$PYTHON_BIN")"

"$PYTHON_BIN" -m venv "$VENV_DIR" --system-site-packages
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_DIR/bin/python" -m pip install -r "$PROJECT_DIR/requirements-rpi.txt"

if ! "$VENV_DIR/bin/python" -c "import torch" >/dev/null 2>&1; then
    echo "PyTorch is missing in this environment."
    echo "Trying a direct pip install for torch..."

    if ! "$VENV_DIR/bin/python" -m pip install torch; then
        echo
        echo "Could not install torch automatically."
        echo "Install a Raspberry Pi compatible torch wheel, then rerun:"
        echo "  $VENV_DIR/bin/python -m pip install /path/to/torch.whl"
        exit 1
    fi
fi

chmod +x "$PROJECT_DIR/start_local_preview.sh"

if [ "$INSTALL_AUTOSTART" -eq 1 ]; then
    mkdir -p "$AUTOSTART_DIR"
    cat >"$AUTOSTART_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=Vehicle Detection Local Preview
Comment=Start the local Raspberry Pi vehicle detection preview
Exec=$PROJECT_DIR/start_local_preview.sh
Path=$PROJECT_DIR
Terminal=false
X-GNOME-Autostart-enabled=true
EOF
fi

echo
echo "Install complete."
echo "Run the app with:"
echo "  $VENV_DIR/bin/python $PROJECT_DIR/main.py"

if [ "$INSTALL_AUTOSTART" -eq 1 ]; then
    echo
    echo "Autostart installed at:"
    echo "  $AUTOSTART_FILE"
fi

if [ ! -f "$PROJECT_DIR/best.pt" ]; then
    echo
    echo "Warning: best.pt is missing. Copy your trained model into the project root."
fi

if [ ! -f "$PROJECT_DIR/rpi-migration/yolo11n.pt" ]; then
    echo
    echo "Warning: rpi-migration/yolo11n.pt is missing. The migration profile will not be available."
fi

echo
echo "After the app starts, open the web view from another device with:"
echo "  http://<pi-ip>:5000"
