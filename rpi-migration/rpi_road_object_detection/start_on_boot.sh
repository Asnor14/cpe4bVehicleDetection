#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/raspi/Downloads/rpi-migration/rpi_road_object_detection"
CAMERA_SOURCE="${CAMERA_SOURCE:-usb}"

cd "$PROJECT_DIR"
exec "$PROJECT_DIR/.venv/bin/python" TFLite_detection_webcam_loop.py \
  --modeldir=TFLite_model_bbd \
  --camera-source="$CAMERA_SOURCE"
