# Raspberry Pi Migration Guide

This folder contains a Raspberry Pi tuned version of the webcam vehicle detector.

## Files
- `main_rpi.py`: optimized script for Raspberry Pi 4B
- `requirements-rpi.txt`: base Python dependencies

## Recommended target
- Raspberry Pi 4B (4GB)
- Raspberry Pi OS 64-bit (Bookworm)
- Python 3.10+

## 1. Copy project to Pi
Use `scp`, USB, or git clone. Make sure `yolo11n.pt` is in the same directory as `main_rpi.py`.

## 2. Create virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 3. Install dependencies
```bash
pip install -r requirements-rpi.txt
```

Install PyTorch separately for your Pi OS/Python combination using a compatible wheel.
Then verify:
```bash
python -c "import torch; print(torch.__version__)"
```

## 4. Run
```bash
python main_rpi.py
```

Press `q` to exit.

If you want to watch the same detections from a pad or phone, use the main app instead:

```bash
python ../main.py
```

That main preview now includes a lightweight web view at `http://<pi-ip>:5000`. `localhost:5000` only works on the Pi itself.

If you are cloning this project onto another Raspberry Pi, you can use the root installer:

```bash
cd ..
./install_on_rpi.sh
```

The migration script now behaves as plain vehicle detection:
- no line counter
- no per-class names like `car` or `bus`
- every detected road vehicle is labeled as `Vehicle`

## Performance notes
- Uses `yolo11n.pt` for lower latency.
- Uses camera resolution `640x360`.
- Uses `imgsz=512` for stronger small-vehicle and motorcycle detection.
- Uses `conf=0.15` to keep motorcycles easier to catch.
- Uses `FRAME_SKIP = 1` so every frame is checked.

## Tuning knobs in `main_rpi.py`
- `FRAME_SKIP`: raise to `2` if you need more smoothness.
- `IMG_SIZE`: reduce to `416` or `320` for faster inference.
- `DETECT_CONF`: raise this if you want fewer weak detections.
- `FRAME_WIDTH`/`FRAME_HEIGHT`: reduce for more FPS.
