# Vehicle Detection Local Preview

This project now runs as a local Raspberry Pi camera preview with a lightweight web viewer.

The old heavier Flask dashboard has been removed so the Pi can focus its CPU on:

- reading the camera
- running YOLO
- drawing the preview window
- serving a low-rate browser stream from the same annotated frame

## Files

- `main.py` - local OpenCV preview app with background capture and inference
- `best.pt` - your trained YOLO model
- `requirements.txt` - Python dependencies
- `requirements-rpi.txt` - Raspberry Pi install dependency list
- `install_on_rpi.sh` - one-step installer for another Raspberry Pi

Keep `best.pt` in the same folder as `main.py`.

## Install

Recommended on Raspberry Pi:

```bash
chmod +x install_on_rpi.sh start_local_preview.sh
./install_on_rpi.sh
```

Manual install if you prefer:

```bash
python3 -m venv .runtime-venv --system-site-packages
source .runtime-venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-rpi.txt
```

If `torch` is still missing after the manual install, install a Raspberry Pi compatible wheel and rerun the command above.

## Push To GitHub

If you want the second Raspberry Pi to get everything from GitHub, you need to `git add`, `git commit`, and `git push`. Editing files locally is not enough by itself.

Push the project code and setup files:

- `main.py`
- `start_local_preview.sh`
- `install_on_rpi.sh`
- `requirements.txt`
- `requirements-rpi.txt`
- `README.md`
- `rpi-migration/`

Do not push local runtime folders like `.runtime-venv/`.

If your model files are small enough for GitHub, you can also push:

- `best.pt`
- `rpi-migration/yolo11n.pt`

If the model files are too large for normal GitHub, copy them to the second Pi separately after cloning.

## Clone On Another Pi

On the other Raspberry Pi:

```bash
git clone <your-repo-url>
cd cpe4BVehicleDetection
chmod +x install_on_rpi.sh start_local_preview.sh
./install_on_rpi.sh
```

If you also want it to start automatically when the Pi desktop logs in:

```bash
./install_on_rpi.sh --autostart
```

The installer will:

- create `.runtime-venv`
- install the Python packages from `requirements-rpi.txt`
- try to install `torch` if it is missing
- create the autostart entry when you use `--autostart`

If `torch` cannot be installed automatically, install a Raspberry Pi compatible wheel first, then rerun `./install_on_rpi.sh`.

## Run

```bash
source .runtime-venv/bin/activate
python3 main.py
```

This opens a native preview window on the Raspberry Pi display and also starts a lightweight web preview on port `5000`.

Open the stream from another device on the same network with:

```text
http://<pi-ip>:5000
```

`localhost:5000` only works on the Raspberry Pi itself. Your pad needs the Pi LAN IP instead.

If `rpi-migration/yolo11n.pt` is present, pressing `C` now switches between the normal local profile and the `RPi Migration` profile. The migration profile keeps its Raspberry Pi tuned vehicle detection settings, but it renders through the same main preview UI.

## Start Automatically On Login

The app is configured to start automatically when the `raspi` desktop session opens.

Installed autostart file:

- `/home/raspi/.config/autostart/vehicle-detection-local.desktop`

Launcher script:

- `start_local_preview.sh`

Autostart desktop template in the repo:

- `vehicle-detection-local.desktop`

Runtime environment used by autostart:

- `.runtime-venv`

Installer for another Pi:

- `install_on_rpi.sh`

Autostart log:

- `/home/raspi/.local/state/vehicle-detection/autostart.log`

Saved startup model:

- `/home/raspi/.config/vehicle-detection/selected-model.txt`

To disable autostart later:

```bash
rm /home/raspi/.config/autostart/vehicle-detection-local.desktop
```

## Controls

- `Q` or `Esc` closes the preview
- `C` switches to the next runtime profile and reloads detection live
- `M` toggles between `predict` mode and `track` mode live
- `-` lowers the confidence threshold
- `+` raises the confidence threshold
- `[` lowers the inference gap for faster tracking
- `]` raises the inference gap for lighter CPU usage
- `R` reopens the camera

If the loaded profile is the single-class custom vehicle model like `best.pt`, the app auto-starts with a tuned threshold of `0.60` and prefers `predict` mode for cleaner CPU-side handling. The `RPi Migration` profile instead uses a lower confidence threshold, larger image size, generic `Vehicle` labels, and COCO vehicle classes so motorcycles are easier to catch.

## Performance Notes

The local preview is tuned for smoother Raspberry Pi performance:

- only one YOLO inference pipeline runs
- YOLO runs in a background worker
- camera capture runs in its own thread so the latest frame stays fresh
- default YOLO image size is `512`
- capture size is `640x480`
- preview refresh is capped at `30 FPS`
- the browser stream reuses the already-annotated frame, caps itself to `6 FPS`, and downscales JPEG output to keep CPU use low
- single-class custom models default to `predict` mode for a lighter and more accurate CPU path
- the `RPi Migration` profile uses `imgsz=512`, `conf=0.15`, generic `Vehicle` labels, and COCO vehicle classes while staying on the same main UI
- `track` mode is still available when you want stable IDs on moving vehicles
- detections stay visible briefly between misses to help with fast-moving vehicles
- the model is fused and warmed up once at startup to reduce first-frame latency
- inference is rate-limited lightly by default, but you can tune it live with `[` and `]`

If you want even more FPS, try:

```bash
python3 main.py --imgsz 416 --inference-interval 0.05 --max-display-fps 30
```

If you want a little more accuracy and can accept lower FPS, try:

```bash
python3 main.py --imgsz 640 --threshold 0.40 --inference-interval 0.00 --max-display-fps 30
```

## CLI Options

```bash
python3 main.py --help
```

Available tuning options include:

- `--model`
- `--camera-index`
- `--threshold`
- `--imgsz`
- `--capture-width`
- `--capture-height`
- `--inference-interval`
- `--inference-mode`
- `--iou-threshold`
- `--max-detections`
- `--skip-model-warmup`
- `--max-display-fps`
- `--disable-web`
- `--headless`
- `--web-host`
- `--web-port`
- `--web-max-fps`
- `--web-width`
- `--web-jpeg-quality`
# raspitwo
