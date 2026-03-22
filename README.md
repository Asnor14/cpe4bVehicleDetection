# Vehicle Warning Dashboard

Simple Python web app for live vehicle detection using an Ultralytics YOLO model saved as `best.pt`.

The dashboard shows two panels:

- `RASPI1` uses camera index `0` by default
- `RASPI2` is shown as an offline placeholder by default and can be enabled later

When a vehicle is detected with confidence at or above the panel threshold:

- the page shows `Vehicle Detected`
- a 5-second countdown appears
- if another qualifying detection happens before the timer ends, the timer resets back to 5 seconds

## Project Files

- `main.py` - Flask web app, YOLO detection, and webcam handling
- `templates/index.html` - dashboard page
- `static/style.css` - page styling
- `best.pt` - your trained YOLO model
- `requirements.txt` - Python dependencies

Keep `best.pt` in the same folder as `main.py`.

## 1. Install Dependencies

Create and activate a virtual environment if you want:

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

Raspberry Pi / Linux:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Run on Windows PC

Make sure your webcam is connected and `best.pt` is in this folder, then run:

```bash
python main.py
```

Open this in your browser on the same PC:

```text
http://localhost:5000
```

To open it from another device on the same network, use:

```text
http://YOUR_PC_IP:5000
```

## 3. Run on Raspberry Pi

Copy the same project folder to the Raspberry Pi so the file layout stays the same:

```text
project_folder/
  main.py
  best.pt
  requirements.txt
  templates/
    index.html
  static/
    style.css
```

Then run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 main.py
```

Open this in a browser on the Pi:

```text
http://localhost:5000
```

Open this from another device on the same network:

```bash
hostname -I
```

Then use the Pi IP, for example:

```text
http://192.168.1.23:5000
```

## Configuration Notes

- Change `DEVICE_CONFIGS` near the top of `main.py` to adjust camera indexes and enabled panels.
- `RASPI1` is enabled by default.
- `RASPI2` is disabled by default, so the second panel stays ready for later.
- Change `DEFAULT_CONFIDENCE` to change the starting threshold.
- You can also edit the threshold live from the browser for each panel.
- The model runs on CPU for compatibility with both Windows and Raspberry Pi.
- `IMAGE_SIZE`, `FRAME_WIDTH`, and `FRAME_HEIGHT` are kept modest to be more Raspberry Pi friendly.
