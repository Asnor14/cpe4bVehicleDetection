import argparse
import contextlib
import html
import os
import socket
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# OpenCV wheels commonly bundle xcb but not the Wayland Qt plugin.
if os.environ.get("XDG_SESSION_TYPE") == "wayland" and not os.environ.get("QT_QPA_PLATFORM"):
    os.environ["QT_QPA_PLATFORM"] = "xcb"
if not os.environ.get("QT_QPA_FONTDIR") and os.path.isdir("/usr/share/fonts/truetype/dejavu"):
    os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts/truetype/dejavu"

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import torch
except Exception:  # pragma: no cover - torch should normally be present
    torch = None


DEFAULT_MODEL_NAME = "best.pt"
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
IMAGE_SIZE = 512
DEFAULT_CONFIDENCE = 0.45
SINGLE_CLASS_MODEL_CONFIDENCE = 0.60
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_MAX_DETECTIONS = 24
DEFAULT_FRAME_SKIP = 1
WARNING_SECONDS = 5.0
INFERENCE_INTERVAL_SECONDS = 0.02
BOX_PERSIST_SECONDS = 0.35
DISPLAY_MAX_FPS = 30.0
TRACKER_CONFIG = "bytetrack.yaml"
AUTO_INFERENCE_MODE = "auto"
PREDICT_INFERENCE_MODE = "predict"
TRACK_INFERENCE_MODE = "track"
DEFAULT_WEB_HOST = "0.0.0.0"
DEFAULT_WEB_PORT = 5000
DEFAULT_WEB_MAX_FPS = 6.0
DEFAULT_WEB_WIDTH = 640
DEFAULT_WEB_JPEG_QUALITY = 70
WINDOW_NAME = "Vehicle Detection Preview"
SETTINGS_DIR = Path.home() / ".config" / "vehicle-detection"
SELECTED_MODEL_FILE = SETTINGS_DIR / "selected-model.txt"
MIGRATION_MODEL_PATH = Path("rpi-migration") / "yolo11n.pt"
MIGRATION_PROFILE_KEY = "rpi-migration"
MIGRATION_PROFILE_LABEL = "RPi Migration"
MIGRATION_DEFAULT_CONFIDENCE = 0.15
MIGRATION_DEFAULT_IMAGE_SIZE = 512
MIGRATION_FRAME_SKIP = 1
MIGRATION_VEHICLE_CLASS_IDS = (1, 2, 3, 5, 6, 7)

VEHICLE_KEYWORDS = {
    "bicycle",
    "bike",
    "bus",
    "car",
    "jeep",
    "motorbike",
    "motorcycle",
    "pickup",
    "truck",
    "tricycle",
    "vehicle",
    "van",
}


def configure_runtime() -> None:
    """Apply a few safe CPU-side optimizations for Raspberry Pi use."""
    cv2.setUseOptimized(True)

    if hasattr(cv2, "setNumThreads"):
        cv2.setNumThreads(max(1, min(2, os.cpu_count() or 1)))

    if torch is not None:
        thread_count = max(1, min(4, os.cpu_count() or 1))
        torch.set_num_threads(thread_count)

        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass


def is_vehicle_name(class_name: str) -> bool:
    """Return True when a class name looks vehicle-related."""
    name = class_name.strip().lower()
    return any(keyword in name for keyword in VEHICLE_KEYWORDS)


def normalize_names(raw_names) -> dict[int, str]:
    """Convert model class names into a simple dictionary."""
    if isinstance(raw_names, dict):
        return {int(class_id): str(class_name) for class_id, class_name in raw_names.items()}

    if isinstance(raw_names, (list, tuple)):
        return {index: str(class_name) for index, class_name in enumerate(raw_names)}

    return {}


def get_vehicle_class_ids(names: dict[int, str]) -> set[int]:
    """Build a set of class IDs that look like vehicle classes."""
    return {
        int(class_id)
        for class_id, class_name in names.items()
        if is_vehicle_name(str(class_name))
    }


def clamp_threshold(value: float) -> float:
    """Keep the confidence threshold between 0.01 and 1.00."""
    return max(0.01, min(1.00, float(value)))


def clamp_iou_threshold(value: float) -> float:
    """Keep the IoU threshold in a safe NMS range."""
    return max(0.05, min(0.95, float(value)))


def clamp_max_detections(value: int) -> int:
    """Keep the detection cap in a practical range."""
    return max(1, min(256, int(value)))


def get_next_inference_mode(current_mode: str) -> str:
    """Toggle between predict and track while the preview is running."""
    return TRACK_INFERENCE_MODE if current_mode == PREDICT_INFERENCE_MODE else PREDICT_INFERENCE_MODE


@dataclass(frozen=True)
class RuntimeProfile:
    key: str
    label: str
    model_path: Path
    default_threshold: float | None = None
    default_image_size: int = IMAGE_SIZE
    default_inference_mode: str = AUTO_INFERENCE_MODE
    frame_skip: int = DEFAULT_FRAME_SKIP
    forced_vehicle_class_ids: tuple[int, ...] = ()
    enable_line_counter: bool = False
    generic_vehicle_label: bool = False


def build_runtime_profile(model_path: Path, project_dir: Path) -> RuntimeProfile:
    """Attach profile-specific behavior to a model path."""
    resolved_model_path = resolve_model_path(model_path, project_dir)
    migration_model_path = (project_dir / MIGRATION_MODEL_PATH).resolve()

    if migration_model_path.exists() and (
        resolved_model_path.resolve() == migration_model_path
        or resolved_model_path.name.lower() == migration_model_path.name.lower()
    ):
        return RuntimeProfile(
            key=MIGRATION_PROFILE_KEY,
            label=MIGRATION_PROFILE_LABEL,
            model_path=migration_model_path,
            default_threshold=MIGRATION_DEFAULT_CONFIDENCE,
            default_image_size=MIGRATION_DEFAULT_IMAGE_SIZE,
            default_inference_mode=PREDICT_INFERENCE_MODE,
            frame_skip=MIGRATION_FRAME_SKIP,
            forced_vehicle_class_ids=MIGRATION_VEHICLE_CLASS_IDS,
            enable_line_counter=False,
            generic_vehicle_label=True,
        )

    return RuntimeProfile(
        key=resolved_model_path.name,
        label=resolved_model_path.name,
        model_path=resolved_model_path,
    )


def get_available_profiles(project_dir: Path) -> list[RuntimeProfile]:
    """Return the switchable runtime profiles for the local preview."""
    migration_model_path = project_dir / MIGRATION_MODEL_PATH
    root_models = []

    for path in project_dir.glob("*.pt"):
        if not path.is_file():
            continue

        # When the migration profile exists, prefer it over the raw root yolo11n.pt entry.
        if migration_model_path.is_file() and path.name.lower() == migration_model_path.name.lower():
            continue

        root_models.append(build_runtime_profile(path, project_dir))

    if migration_model_path.is_file():
        root_models.append(build_runtime_profile(migration_model_path, project_dir))

    return sorted(
        root_models,
        key=lambda profile: (
            profile.model_path.name.lower() != DEFAULT_MODEL_NAME.lower(),
            profile.key == MIGRATION_PROFILE_KEY,
            str(profile.model_path).lower(),
        ),
    )


def resolve_model_path(model_value: str | Path | None, project_dir: Path) -> Path:
    """Resolve a model path against the project directory."""
    if model_value in (None, ""):
        return project_dir / DEFAULT_MODEL_NAME

    model_path = Path(model_value)
    if not model_path.is_absolute():
        model_path = project_dir / model_path

    return model_path


def read_selected_model(project_dir: Path) -> Path | None:
    """Read the saved startup model if it still exists."""
    if not SELECTED_MODEL_FILE.exists():
        return None

    try:
        raw_value = SELECTED_MODEL_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return None

    if not raw_value:
        return None

    model_path = resolve_model_path(raw_value, project_dir)
    return model_path if model_path.exists() else None


def save_selected_model(project_dir: Path, model_path: Path) -> None:
    """Persist the selected model so autostart can reuse it."""
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        stored_value = str(model_path.relative_to(project_dir))
    except ValueError:
        stored_value = str(model_path)

    try:
        SELECTED_MODEL_FILE.write_text(stored_value, encoding="utf-8")
    except OSError as error:
        print(f"Warning: could not save the selected model. Details: {error}")


def get_startup_model_path(project_dir: Path, requested_model: str | None) -> Path:
    """Pick a startup model from CLI, saved config, or the default file."""
    if requested_model:
        return resolve_model_path(requested_model, project_dir)

    saved_model_path = read_selected_model(project_dir)
    if saved_model_path is not None:
        return saved_model_path

    return project_dir / DEFAULT_MODEL_NAME


def get_next_profile(current_model_path: Path, available_profiles: list[RuntimeProfile]) -> RuntimeProfile | None:
    """Cycle to the next available runtime profile."""
    if not available_profiles:
        return None

    resolved_current = current_model_path.resolve()

    for index, profile in enumerate(available_profiles):
        if profile.model_path.resolve() == resolved_current:
            return available_profiles[(index + 1) % len(available_profiles)]

    return available_profiles[0]


def track_color(track_id: int) -> tuple[int, int, int]:
    """Generate a stable color for a tracked vehicle ID."""
    palette = [
        (48, 219, 91),
        (112, 216, 255),
        (255, 190, 92),
        (255, 107, 129),
        (165, 94, 234),
        (255, 159, 243),
        (0, 210, 211),
        (255, 127, 80),
    ]
    return palette[track_id % len(palette)]


def open_camera(camera_index: int, width: int, height: int):
    """Open the camera with a low-latency configuration."""
    if os.name == "nt":
        attempts = [
            getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
            cv2.CAP_ANY,
            getattr(cv2, "CAP_MSMF", cv2.CAP_ANY),
        ]
    else:
        attempts = [getattr(cv2, "CAP_V4L2", cv2.CAP_ANY), cv2.CAP_ANY]

    mjpg_fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    for backend in attempts:
        cap = cv2.VideoCapture(camera_index, backend)
        if not cap.isOpened():
            cap.release()
            continue

        if hasattr(cv2, "CAP_PROP_FOURCC"):
            cap.set(cv2.CAP_PROP_FOURCC, mjpg_fourcc)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if hasattr(cv2, "CAP_PROP_FPS"):
            cap.set(cv2.CAP_PROP_FPS, 30)

        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        return cap

    return None


def build_placeholder_frame(width: int, height: int, title: str, message: str):
    """Create a simple placeholder frame for camera or model errors."""
    frame = np.full((height, width, 3), (236, 236, 236), dtype=np.uint8)
    cv2.rectangle(frame, (10, 10), (width - 10, height - 10), (32, 32, 32), 2)
    cv2.putText(
        frame,
        title,
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (32, 32, 32),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        message,
        (20, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (88, 88, 88),
        2,
        cv2.LINE_AA,
    )
    return frame


def encode_jpeg_frame(frame, jpeg_quality: int) -> bytes:
    """Encode a frame once so the web view can reuse the same JPEG bytes."""
    success, buffer = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), max(30, min(95, int(jpeg_quality)))],
    )
    if not success:
        raise RuntimeError("Could not encode the web preview frame.")
    return buffer.tobytes()


def get_local_ip_address() -> str | None:
    """Best-effort LAN IP for showing the tablet-friendly URL."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("10.255.255.255", 1))
            return sock.getsockname()[0]
    except OSError:
        return None


def has_local_display() -> bool:
    """Return True when the Pi session has a display we can render to."""
    if os.name == "nt":
        return True

    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class WebPreviewServer:
    """Serve a low-rate MJPEG stream using the already-annotated preview frame."""

    def __init__(
        self,
        host: str,
        port: int,
        max_fps: float,
        width: int,
        jpeg_quality: int,
        placeholder_frame,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.max_fps = max(1.0, float(max_fps))
        self.frame_interval = 1.0 / self.max_fps
        self.width = max(160, int(width))
        self.jpeg_quality = max(30, min(95, int(jpeg_quality)))
        self.condition = threading.Condition()
        self.latest_jpeg = encode_jpeg_frame(placeholder_frame, self.jpeg_quality)
        self.frame_serial = 0
        self.last_encoded_at = 0.0
        self.active_stream_clients = 0
        self.httpd = None
        self.thread = None
        self.local_ip = get_local_ip_address()

    def start(self) -> None:
        if self.thread is not None:
            return

        handler_cls = self._build_handler_class()
        class PreviewHTTPServer(ThreadingHTTPServer):
            allow_reuse_address = True
            daemon_threads = True

        self.httpd = PreviewHTTPServer((self.host, self.port), handler_cls)
        self.port = int(self.httpd.server_address[1])
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None

        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None

    def update_frame(self, frame) -> None:
        """Encode at a capped rate so web viewing does not add much CPU load."""
        now = time.perf_counter()

        with self.condition:
            idle_refresh_interval = 2.0
            if self.active_stream_clients == 0 and now - self.last_encoded_at < idle_refresh_interval:
                return

            if now - self.last_encoded_at < self.frame_interval:
                return

        if frame.shape[1] > self.width:
            scaled_height = max(1, int(frame.shape[0] * (self.width / frame.shape[1])))
            web_frame = cv2.resize(frame, (self.width, scaled_height), interpolation=cv2.INTER_AREA)
        else:
            web_frame = frame

        jpeg_bytes = encode_jpeg_frame(web_frame, self.jpeg_quality)

        with self.condition:
            self.latest_jpeg = jpeg_bytes
            self.last_encoded_at = now
            self.frame_serial += 1
            self.condition.notify_all()

    def wait_for_frame(self, previous_serial: int, timeout: float = 1.0) -> tuple[bytes, int]:
        """Block briefly until a new JPEG is ready, then return the latest bytes."""
        with self.condition:
            if self.frame_serial == previous_serial:
                self.condition.wait(timeout=timeout)

            return self.latest_jpeg, self.frame_serial

    def stream_started(self) -> None:
        with self.condition:
            self.active_stream_clients += 1

    def stream_stopped(self) -> None:
        with self.condition:
            self.active_stream_clients = max(0, self.active_stream_clients - 1)

    def local_url(self) -> str:
        return f"http://localhost:{self.port}"

    def network_url(self) -> str:
        if self.local_ip:
            return f"http://{self.local_ip}:{self.port}"
        return f"http://<pi-ip>:{self.port}"

    def _build_handler_class(self):
        preview_server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
                if self.path in ("/", "/index.html"):
                    page = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Vehicle Detection Web View</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #111; color: #f4f4f4; }}
    main {{ max-width: 980px; margin: 0 auto; padding: 16px; }}
    .card {{ background: #1c1c1c; border-radius: 12px; overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,0.35); }}
    img {{ display: block; width: 100%; height: auto; background: #000; }}
    p {{ margin: 12px 0 0; line-height: 1.5; color: #d7d7d7; }}
    code {{ color: #9fe870; }}
  </style>
</head>
<body>
  <main>
    <div class=\"card\"><img src=\"/stream.mjpg\" alt=\"Vehicle preview\"></div>
    <p>Open <code>{html.escape(preview_server.network_url())}</code> from your pad on the same network.</p>
    <p><code>localhost</code> only works on the Raspberry Pi itself.</p>
  </main>
</body>
</html>"""
                    encoded = page.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(encoded)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(encoded)
                    return

                if self.path == "/snapshot.jpg":
                    jpeg_bytes, _serial = preview_server.wait_for_frame(-1, timeout=0.0)
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpeg_bytes)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(jpeg_bytes)
                    return

                if self.path != "/stream.mjpg":
                    self.send_error(404)
                    return

                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()

                preview_server.stream_started()
                serial = -1
                try:
                    while True:
                        jpeg_bytes, serial = preview_server.wait_for_frame(serial)
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg_bytes)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(jpeg_bytes)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass
                finally:
                    preview_server.stream_stopped()

            def log_message(self, format, *args):  # noqa: A003 - BaseHTTPRequestHandler API
                return

        return Handler


@dataclass
class LoadedModelBundle:
    model: object | None
    model_lock: threading.Lock
    class_names: dict[int, str] = field(default_factory=dict)
    vehicle_class_ids: set[int] = field(default_factory=set)
    model_error: str | None = None
    single_vehicle_class: bool = False


def is_single_vehicle_model(class_names: dict[int, str], vehicle_class_ids: set[int]) -> bool:
    """Detect the common custom-model case where one class simply means vehicle."""
    return len(class_names) == 1 and len(vehicle_class_ids) == 1


def resolve_inference_mode(requested_mode: str, model_bundle: LoadedModelBundle) -> str:
    """Pick the runtime inference mode for the loaded model."""
    if requested_mode != AUTO_INFERENCE_MODE:
        return requested_mode

    if model_bundle.single_vehicle_class:
        return PREDICT_INFERENCE_MODE

    return TRACK_INFERENCE_MODE


def resolve_starting_threshold(requested_threshold: float | None, model_bundle: LoadedModelBundle) -> float:
    """Use a slightly stricter default for single-class custom vehicle models."""
    if requested_threshold is not None:
        return clamp_threshold(requested_threshold)

    if model_bundle.single_vehicle_class:
        return SINGLE_CLASS_MODEL_CONFIDENCE

    return DEFAULT_CONFIDENCE


def resolve_profile_inference_mode(
    requested_mode: str,
    model_bundle: LoadedModelBundle,
    profile: RuntimeProfile,
) -> str:
    """Pick inference mode, letting a profile override the generic auto behavior."""
    if requested_mode != AUTO_INFERENCE_MODE:
        return requested_mode

    if profile.default_inference_mode != AUTO_INFERENCE_MODE:
        return profile.default_inference_mode

    return resolve_inference_mode(requested_mode, model_bundle)


def resolve_profile_threshold(
    requested_threshold: float | None,
    model_bundle: LoadedModelBundle,
    profile: RuntimeProfile,
) -> float:
    """Pick a startup threshold, preferring profile defaults when available."""
    if requested_threshold is not None:
        return clamp_threshold(requested_threshold)

    if profile.default_threshold is not None:
        return clamp_threshold(profile.default_threshold)

    return resolve_starting_threshold(requested_threshold, model_bundle)


def resolve_profile_image_size(requested_image_size: int | None, profile: RuntimeProfile) -> int:
    """Pick an image size, allowing profile defaults like the migration path."""
    if requested_image_size is not None:
        return max(32, int(requested_image_size))

    return max(32, int(profile.default_image_size))


def resolve_profile_vehicle_class_ids(
    profile: RuntimeProfile,
    model_bundle: LoadedModelBundle,
) -> set[int]:
    """Use profile-specific class IDs when the migration profile needs them."""
    if profile.forced_vehicle_class_ids:
        return {int(class_id) for class_id in profile.forced_vehicle_class_ids}

    return set(model_bundle.vehicle_class_ids)


def resolve_profile_class_names(
    profile: RuntimeProfile,
    model_bundle: LoadedModelBundle,
) -> dict[int, str]:
    """Override class labels when a profile should render all hits as Vehicle."""
    if profile.generic_vehicle_label:
        vehicle_class_ids = resolve_profile_vehicle_class_ids(profile, model_bundle)
        return {class_id: "Vehicle" for class_id in vehicle_class_ids}

    return dict(model_bundle.class_names)


def optimize_loaded_model(model) -> None:
    """Apply lightweight YOLO inference optimizations when supported."""
    try:
        model.fuse()
    except Exception:
        pass


def warmup_loaded_model(
    model_bundle: LoadedModelBundle,
    image_size: int,
    iou_threshold: float,
) -> None:
    """Prime the model once so the first live frame is less expensive."""
    if model_bundle.model is None:
        return

    warmup_frame = np.zeros((CAPTURE_HEIGHT, CAPTURE_WIDTH, 3), dtype=np.uint8)
    predict_kwargs = {
        "source": warmup_frame,
        "conf": 0.25,
        "iou": clamp_iou_threshold(iou_threshold),
        "imgsz": max(32, int(image_size)),
        "device": "cpu",
        "verbose": False,
        "max_det": 1,
    }

    if model_bundle.vehicle_class_ids:
        predict_kwargs["classes"] = sorted(model_bundle.vehicle_class_ids)

    inference_context = torch.inference_mode() if torch is not None else contextlib.nullcontext()

    try:
        with model_bundle.model_lock:
            with inference_context:
                model_bundle.model.predict(**predict_kwargs)
    except Exception as error:
        print(f"Warning: model warmup skipped. Details: {error}")


def load_model(model_path: Path) -> LoadedModelBundle:
    """Load the YOLO model, class metadata, and tuned defaults once."""
    model = None
    model_error = None
    class_names = {}
    vehicle_class_ids = set()
    model_lock = threading.Lock()

    if not model_path.exists():
        model_error = f"Error: '{model_path.name}' was not found in the project folder."
        print(model_error)
    else:
        try:
            model = YOLO(str(model_path))
            optimize_loaded_model(model)
            class_names = normalize_names(getattr(model, "names", None))
            vehicle_class_ids = get_vehicle_class_ids(class_names)

            if vehicle_class_ids:
                print("Vehicle class filter enabled.")
            else:
                print("Note: no vehicle-like class names were found, so all detections will be shown.")
        except Exception as error:
            model_error = f"Error: failed to load model '{model_path.name}'. Details: {error}"
            print(model_error)

    return LoadedModelBundle(
        model=model,
        model_lock=model_lock,
        class_names=class_names,
        vehicle_class_ids=vehicle_class_ids,
        model_error=model_error,
        single_vehicle_class=is_single_vehicle_model(class_names, vehicle_class_ids),
    )


class CameraStream:
    """Continuously read the newest frame so display stays responsive."""

    def __init__(self, camera_index: int, width: int, height: int) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None
        self.thread = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_frame_id = 0
        self.last_error = ""
        self.camera_open = False

    def start(self) -> None:
        if self.thread is not None:
            return

        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def restart(self) -> None:
        with self.lock:
            self.last_error = ""
        self._release_camera()

    def stop(self) -> None:
        self.stop_event.set()

        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None

        self._release_camera()

    def get_latest_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return 0, None

            return self.latest_frame_id, self.latest_frame.copy()

    def get_camera_state(self):
        with self.lock:
            return self.camera_open, self.last_error

    def _run_loop(self) -> None:
        while not self.stop_event.is_set():
            if self.cap is None:
                self.cap = open_camera(self.camera_index, self.width, self.height)

                if self.cap is None:
                    self._set_error(f"Could not open webcam at camera index {self.camera_index}.")
                    time.sleep(1.0)
                    continue

                with self.lock:
                    self.camera_open = True
                    self.last_error = ""

            success, frame = self.cap.read()

            if not success or frame is None:
                self._set_error("Failed to read a frame from the webcam.")
                self._release_camera()
                time.sleep(0.25)
                continue

            with self.lock:
                self.latest_frame = frame
                self.latest_frame_id += 1

    def _set_error(self, message: str) -> None:
        with self.lock:
            self.camera_open = False
            self.last_error = message

    def _release_camera(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        with self.lock:
            self.camera_open = False


@dataclass
class DetectionBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    track_id: int | None = None
    class_id: int | None = None
    class_name: str = "Vehicle"


@dataclass
class DetectionSnapshot:
    boxes: list[DetectionBox] = field(default_factory=list)
    best_confidence: float = 0.0
    last_boxes_at: float = 0.0
    warning_until: float = 0.0
    status_message: str = "Starting"
    last_error: str = ""
    inference_ms: float = 0.0
    threshold: float = DEFAULT_CONFIDENCE
    class_counts: dict[str, int] = field(default_factory=dict)
    count_line_y: int | None = None


class InferenceWorker:
    """Run YOLO in the background on the newest available frame."""

    def __init__(
        self,
        camera_stream: CameraStream,
        model,
        model_lock: threading.Lock,
        class_names: dict[int, str],
        vehicle_class_ids: set[int],
        threshold: float,
        image_size: int,
        inference_interval: float,
        inference_mode: str,
        iou_threshold: float,
        max_detections: int,
        frame_skip: int,
        count_line_enabled: bool,
    ) -> None:
        self.camera_stream = camera_stream
        self.model = model
        self.model_lock = model_lock
        self.class_names = class_names
        self.vehicle_class_ids = vehicle_class_ids
        self.image_size = image_size
        self.inference_interval = max(0.0, float(inference_interval))
        self.inference_mode = inference_mode
        self.iou_threshold = clamp_iou_threshold(iou_threshold)
        self.max_detections = clamp_max_detections(max_detections)
        self.frame_skip = max(1, int(frame_skip))
        self.count_line_enabled = count_line_enabled
        self.stop_event = threading.Event()
        self.thread = None
        self.lock = threading.Lock()
        self.snapshot = DetectionSnapshot(threshold=clamp_threshold(threshold))
        self.class_counts: dict[str, int] = {}
        self.crossed_track_ids: set[int] = set()
        self.last_processed_frame_id = 0
        self.last_inference_started_at = 0.0

    def start(self) -> None:
        if self.thread is not None:
            return

        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None

    def adjust_threshold(self, delta: float) -> float:
        with self.lock:
            self.snapshot.threshold = clamp_threshold(self.snapshot.threshold + delta)
            return self.snapshot.threshold

    def adjust_inference_interval(self, delta: float) -> float:
        with self.lock:
            self.inference_interval = max(0.0, self.inference_interval + delta)
            return self.inference_interval

    def get_snapshot(self) -> DetectionSnapshot:
        with self.lock:
            return DetectionSnapshot(
                boxes=[
                    DetectionBox(
                        x1=box.x1,
                        y1=box.y1,
                        x2=box.x2,
                        y2=box.y2,
                        confidence=box.confidence,
                        track_id=box.track_id,
                        class_id=box.class_id,
                        class_name=box.class_name,
                    )
                    for box in self.snapshot.boxes
                ],
                best_confidence=self.snapshot.best_confidence,
                last_boxes_at=self.snapshot.last_boxes_at,
                warning_until=self.snapshot.warning_until,
                status_message=self.snapshot.status_message,
                last_error=self.snapshot.last_error,
                inference_ms=self.snapshot.inference_ms,
                threshold=self.snapshot.threshold,
                class_counts=dict(self.snapshot.class_counts),
                count_line_y=self.snapshot.count_line_y,
            )

    def get_inference_interval(self) -> float:
        with self.lock:
            return self.inference_interval

    def get_inference_mode(self) -> str:
        return self.inference_mode

    def _run_loop(self) -> None:
        while not self.stop_event.is_set():
            frame_id, frame = self.camera_stream.get_latest_frame()

            if frame is None:
                time.sleep(0.01)
                continue

            now = time.perf_counter()
            if frame_id == self.last_processed_frame_id:
                time.sleep(0.002)
                continue

            if now - self.last_inference_started_at < self.inference_interval:
                time.sleep(0.002)
                continue

            if self.frame_skip > 1 and frame_id % self.frame_skip != 0:
                time.sleep(0.002)
                continue

            snapshot = self.get_snapshot()
            self.last_inference_started_at = now
            self.last_processed_frame_id = frame_id

            try:
                start = time.perf_counter()
                boxes, best_confidence = detect_vehicle_boxes(
                    frame=frame,
                    model=self.model,
                    model_lock=self.model_lock,
                    class_names=self.class_names,
                    vehicle_class_ids=self.vehicle_class_ids,
                    threshold=snapshot.threshold,
                    image_size=self.image_size,
                    inference_mode=self.inference_mode,
                    iou_threshold=self.iou_threshold,
                    max_detections=self.max_detections,
                )
                inference_ms = (time.perf_counter() - start) * 1000.0

                with self.lock:
                    if boxes:
                        self.snapshot.boxes = boxes
                        self.snapshot.best_confidence = best_confidence
                        self.snapshot.last_boxes_at = time.time()
                    elif time.time() - self.snapshot.last_boxes_at > BOX_PERSIST_SECONDS:
                        self.snapshot.boxes = []
                        self.snapshot.best_confidence = 0.0

                    self.snapshot.inference_ms = inference_ms
                    self.snapshot.last_error = ""
                    if self.count_line_enabled:
                        line_y = frame.shape[0] // 2
                        self._update_line_counts(boxes, line_y)
                        self.snapshot.class_counts = dict(self.class_counts)
                        self.snapshot.count_line_y = line_y
                    else:
                        self.snapshot.class_counts = {}
                        self.snapshot.count_line_y = None

                    if boxes:
                        self.snapshot.warning_until = time.time() + WARNING_SECONDS
                        self.snapshot.status_message = "Vehicle detected"
                    else:
                        if self.snapshot.boxes:
                            self.snapshot.status_message = (
                                "Tracking"
                                if self.inference_mode == TRACK_INFERENCE_MODE
                                else "Holding last detection"
                            )
                        else:
                            self.snapshot.status_message = "Watching"
            except Exception as error:
                with self.lock:
                    self.snapshot.last_error = str(error)
                    self.snapshot.status_message = "Model error"

                time.sleep(0.05)

    def _update_line_counts(self, boxes: list[DetectionBox], line_y: int) -> None:
        """Count tracked objects once after they cross the migration center line."""
        for box in boxes:
            if box.track_id is None:
                continue

            center_y = (box.y1 + box.y2) // 2
            if center_y <= line_y or box.track_id in self.crossed_track_ids:
                continue

            self.crossed_track_ids.add(box.track_id)
            count_label = box.class_name or "vehicle"
            self.class_counts[count_label] = self.class_counts.get(count_label, 0) + 1


def detect_vehicle_boxes(
    frame,
    model,
    model_lock: threading.Lock,
    class_names: dict[int, str],
    vehicle_class_ids: set[int],
    threshold: float,
    image_size: int,
    inference_mode: str,
    iou_threshold: float,
    max_detections: int,
):
    """Run YOLO on one frame and collect vehicle-like boxes."""
    best_confidence = 0.0
    collected_boxes = []

    common_kwargs = {
        "source": frame,
        "conf": threshold,
        "iou": clamp_iou_threshold(iou_threshold),
        "imgsz": image_size,
        "device": "cpu",
        "verbose": False,
        "max_det": clamp_max_detections(max_detections),
    }

    if vehicle_class_ids:
        common_kwargs["classes"] = sorted(vehicle_class_ids)

    inference_context = torch.inference_mode() if torch is not None else contextlib.nullcontext()

    with model_lock:
        with inference_context:
            if inference_mode == TRACK_INFERENCE_MODE:
                results = model.track(
                    **common_kwargs,
                    persist=True,
                    tracker=TRACKER_CONFIG,
                )
            else:
                results = model.predict(**common_kwargs)

    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return collected_boxes, best_confidence

    xyxy_rows = boxes.xyxy.tolist()
    confidence_rows = boxes.conf.tolist()
    class_rows = boxes.cls.tolist() if boxes.cls is not None else []
    track_rows = boxes.id.tolist() if getattr(boxes, "id", None) is not None else [None] * len(xyxy_rows)

    for index, row in enumerate(xyxy_rows):
        class_id = int(class_rows[index]) if class_rows else -1
        confidence = float(confidence_rows[index])

        if vehicle_class_ids and class_id not in vehicle_class_ids:
            continue

        x1, y1, x2, y2 = map(int, row)
        track_id = track_rows[index]
        normalized_track_id = int(track_id) if track_id is not None else None
        class_name = str(class_names.get(class_id, "vehicle")) if class_id >= 0 else "vehicle"

        collected_boxes.append(
            DetectionBox(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=confidence,
                track_id=normalized_track_id,
                class_id=class_id,
                class_name=class_name,
            )
        )
        best_confidence = max(best_confidence, confidence)

    return collected_boxes, best_confidence


def build_inference_worker(
    camera_stream: CameraStream,
    model_bundle: LoadedModelBundle,
    profile: RuntimeProfile,
    class_names: dict[int, str],
    vehicle_class_ids: set[int],
    threshold: float,
    image_size: int,
    inference_interval: float,
    inference_mode: str,
    iou_threshold: float,
    max_detections: int,
    warmup_model: bool,
) -> InferenceWorker:
    """Start a fresh inference worker for the already-loaded model."""
    if model_bundle.model is None:
        raise RuntimeError(model_bundle.model_error or "Failed to load the selected model.")

    if warmup_model:
        warmup_loaded_model(
            model_bundle=model_bundle,
            image_size=image_size,
            iou_threshold=iou_threshold,
        )

    inference_worker = InferenceWorker(
        camera_stream=camera_stream,
        model=model_bundle.model,
        model_lock=model_bundle.model_lock,
        class_names=class_names,
        vehicle_class_ids=vehicle_class_ids,
        threshold=threshold,
        image_size=image_size,
        inference_interval=inference_interval,
        inference_mode=inference_mode,
        iou_threshold=iou_threshold,
        max_detections=max_detections,
        frame_skip=profile.frame_skip,
        count_line_enabled=profile.enable_line_counter,
    )
    inference_worker.start()
    return inference_worker


def draw_preview(
    frame,
    snapshot: DetectionSnapshot,
    camera_open: bool,
    camera_error: str,
    display_fps: float,
    inference_interval: float,
    inference_mode: str,
    image_size: int,
    profile_label: str,
    max_display_fps: float,
):
    """Draw the current detection state on the latest frame."""
    annotated = frame.copy()
    now = time.time()
    warning_active = now < snapshot.warning_until
    countdown = max(0.0, snapshot.warning_until - now)

    active_track_ids = {
        box.track_id
        for box in snapshot.boxes
        if box.track_id is not None
    }

    for box in snapshot.boxes:
        if box.track_id is None:
            color = (48, 219, 91)
            label = f"{box.class_name} {box.confidence:.2f}"
        else:
            color = track_color(box.track_id)
            label = f"ID {box.track_id} {box.class_name}"

        cv2.rectangle(annotated, (box.x1, box.y1), (box.x2, box.y2), color, 2)
        label_y = box.y1 - 10 if box.y1 > 24 else box.y1 + 24
        cv2.putText(
            annotated,
            label,
            (box.x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    header_lines = [
        f"Profile {profile_label}  |  Mode {inference_mode}  |  Threshold {snapshot.threshold:.2f}  |  Display FPS {display_fps:.1f}/{max_display_fps:.0f}",
        f"Inference {snapshot.inference_ms:.0f} ms  |  Img size {image_size}  |  Gap {inference_interval:.2f}s",
        f"Visible vehicles {len(active_track_ids) or len(snapshot.boxes)}  |  Keys: C profile | M mode | -/+ conf | [/] gap | R cam | Q quit",
    ]

    header_height = 82
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], header_height), (245, 245, 245), -1)

    for index, text in enumerate(header_lines):
        cv2.putText(
            annotated,
            text,
            (12, 24 + (index * 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (24, 24, 24),
            1,
            cv2.LINE_AA,
        )

    if snapshot.count_line_y is not None:
        cv2.line(
            annotated,
            (0, snapshot.count_line_y),
            (annotated.shape[1] - 1, snapshot.count_line_y),
            (0, 0, 255),
            2,
        )
        cv2.putText(
            annotated,
            "Count Line",
            (12, max(header_height + 20, snapshot.count_line_y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if snapshot.class_counts:
        y_offset = header_height + 26
        for class_name, count in sorted(snapshot.class_counts.items()):
            cv2.putText(
                annotated,
                f"{class_name}: {count}",
                (12, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (48, 219, 91),
                2,
                cv2.LINE_AA,
            )
            y_offset += 24

    if camera_error:
        footer_text = camera_error
        footer_color = (58, 58, 188)
    elif snapshot.last_error:
        footer_text = snapshot.last_error
        footer_color = (58, 58, 188)
    elif warning_active:
        footer_text = f"Vehicle detected  |  Confidence {snapshot.best_confidence:.2f}  |  Clear in {countdown:.1f}s"
        footer_color = (48, 219, 91)
    elif snapshot.boxes:
        footer_text = f"{snapshot.status_message}  |  Showing {len(active_track_ids) or len(snapshot.boxes)} vehicle(s)"
        footer_color = (112, 216, 255)
    elif camera_open:
        footer_text = "Watching"
        footer_color = (255, 255, 255)
    else:
        footer_text = "Waiting for camera"
        footer_color = (220, 220, 220)

    footer_top = max(0, annotated.shape[0] - 34)
    cv2.rectangle(annotated, (0, footer_top), (annotated.shape[1], annotated.shape[0]), (18, 18, 18), -1)
    cv2.putText(
        annotated,
        footer_text,
        (12, annotated.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        footer_color,
        1,
        cv2.LINE_AA,
    )

    return annotated


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI options for the local Raspberry Pi preview."""
    parser = argparse.ArgumentParser(description="Local Raspberry Pi vehicle detection preview.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model/profile to load, for example best.pt or rpi-migration/yolo11n.pt.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera index to use.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Starting confidence threshold. If omitted, a tuned default is chosen for the loaded model.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="YOLO image size. If omitted, the active profile chooses a tuned default.",
    )
    parser.add_argument(
        "--capture-width",
        type=int,
        default=CAPTURE_WIDTH,
        help="Requested capture width.",
    )
    parser.add_argument(
        "--capture-height",
        type=int,
        default=CAPTURE_HEIGHT,
        help="Requested capture height.",
    )
    parser.add_argument(
        "--inference-interval",
        type=float,
        default=INFERENCE_INTERVAL_SECONDS,
        help="Minimum seconds between inference passes.",
    )
    parser.add_argument(
        "--inference-mode",
        type=str,
        choices=[AUTO_INFERENCE_MODE, PREDICT_INFERENCE_MODE, TRACK_INFERENCE_MODE],
        default=AUTO_INFERENCE_MODE,
        help="Use auto, predict, or track inference.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=DEFAULT_IOU_THRESHOLD,
        help="IoU threshold used during non-max suppression.",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=DEFAULT_MAX_DETECTIONS,
        help="Maximum detections to keep per frame.",
    )
    parser.add_argument(
        "--skip-model-warmup",
        action="store_true",
        help="Skip the one-time model warmup pass at startup and after model switches.",
    )
    parser.add_argument(
        "--max-display-fps",
        type=float,
        default=DISPLAY_MAX_FPS,
        help="Maximum preview refresh rate.",
    )
    parser.add_argument(
        "--disable-web",
        action="store_true",
        help="Turn off the lightweight MJPEG web preview.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without the local OpenCV window and serve only the web preview.",
    )
    parser.add_argument(
        "--web-host",
        type=str,
        default=DEFAULT_WEB_HOST,
        help="Host/interface for the web preview server.",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=DEFAULT_WEB_PORT,
        help="Port for the web preview server.",
    )
    parser.add_argument(
        "--web-max-fps",
        type=float,
        default=DEFAULT_WEB_MAX_FPS,
        help="Maximum FPS for the web preview stream.",
    )
    parser.add_argument(
        "--web-width",
        type=int,
        default=DEFAULT_WEB_WIDTH,
        help="Maximum width for the web preview stream.",
    )
    parser.add_argument(
        "--web-jpeg-quality",
        type=int,
        default=DEFAULT_WEB_JPEG_QUALITY,
        help="JPEG quality for the web preview stream.",
    )
    return parser


def run_local_preview(args) -> int:
    """Open a native OpenCV preview on the Raspberry Pi."""
    display_available = has_local_display()
    web_enabled = not args.disable_web

    if args.headless and not web_enabled:
        print("Error: headless mode needs the web preview enabled.")
        return 1

    if not display_available and not web_enabled:
        print("Error: no local display was found, and the web preview is disabled.")
        return 1

    headless = bool(args.headless or not display_available)
    configure_runtime()

    project_dir = Path(__file__).resolve().parent
    available_profiles = get_available_profiles(project_dir)
    startup_model_path = get_startup_model_path(project_dir, args.model)
    current_profile = build_runtime_profile(startup_model_path, project_dir)
    current_model_bundle = load_model(current_profile.model_path)
    current_threshold = resolve_profile_threshold(args.threshold, current_model_bundle, current_profile)
    current_inference_mode = resolve_profile_inference_mode(args.inference_mode, current_model_bundle, current_profile)
    current_image_size = resolve_profile_image_size(args.imgsz, current_profile)
    current_class_names = resolve_profile_class_names(current_profile, current_model_bundle)
    current_vehicle_class_ids = resolve_profile_vehicle_class_ids(current_profile, current_model_bundle)
    current_iou_threshold = clamp_iou_threshold(args.iou_threshold)
    current_max_detections = clamp_max_detections(args.max_detections)
    manual_mode_override = False
    manual_threshold_override = args.threshold is not None
    placeholder_frame = build_placeholder_frame(
        args.capture_width,
        args.capture_height,
        "RASPI1",
        "Starting preview",
    )

    camera_stream = CameraStream(
        camera_index=args.camera_index,
        width=args.capture_width,
        height=args.capture_height,
    )
    camera_stream.start()

    try:
        inference_worker = build_inference_worker(
            camera_stream=camera_stream,
            model_bundle=current_model_bundle,
            profile=current_profile,
            class_names=current_class_names,
            vehicle_class_ids=current_vehicle_class_ids,
            threshold=current_threshold,
            image_size=current_image_size,
            inference_interval=args.inference_interval,
            inference_mode=current_inference_mode,
            iou_threshold=current_iou_threshold,
            max_detections=current_max_detections,
            warmup_model=not args.skip_model_warmup,
        )
    except RuntimeError as error:
        print(error)
        camera_stream.stop()
        return 1

    save_selected_model(project_dir, current_profile.model_path)

    web_server = None
    if web_enabled:
        try:
            web_server = WebPreviewServer(
                host=args.web_host,
                port=args.web_port,
                max_fps=args.web_max_fps,
                width=args.web_width,
                jpeg_quality=args.web_jpeg_quality,
                placeholder_frame=placeholder_frame,
            )
            web_server.start()
        except OSError as error:
            if headless:
                inference_worker.stop()
                camera_stream.stop()
                print(f"Error: could not start the web preview. Details: {error}")
                return 1

            print(f"Warning: web preview is unavailable. Details: {error}")
            web_server = None

    if not headless:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, args.capture_width, args.capture_height)

    if headless:
        print(f"Opening headless preview for camera index {args.camera_index}.")
    else:
        print(f"Opening local preview for camera index {args.camera_index}. Press Q to quit.")
    print(f"Profile: {current_profile.label}")
    print(
        f"Mode: {current_inference_mode}  |  Threshold: {current_threshold:.2f}  |  "
        f"Img size: {current_image_size}  |  IoU: {current_iou_threshold:.2f}  |  Max detections: {current_max_detections}"
    )
    if web_server is not None:
        print(f"Web preview on the Pi: {web_server.local_url()}")
        print(f"Open on your pad: {web_server.network_url()}")
        print("Note: localhost only works on the Raspberry Pi itself.")
    if args.threshold is None and current_profile.default_threshold is not None:
        print(f"Using {current_profile.label} default threshold {current_threshold:.2f}.")
    elif args.threshold is None and current_model_bundle.single_vehicle_class:
        print(f"Using tuned threshold {current_threshold:.2f} for the single-class vehicle model.")
    if args.inference_mode == AUTO_INFERENCE_MODE:
        print(f"Auto mode selected {current_inference_mode} inference for {current_profile.label}.")
    if headless:
        print("Headless mode is active. Use Ctrl+C in the terminal to stop the app.")
    else:
        print("Use C to switch profiles. Use M to toggle predict/track. Use - and + to change threshold. Use [ and ] to change inference gap. Use R to reopen the camera.")

    last_display_time = time.perf_counter()
    display_fps = 0.0
    display_frame_period = 1.0 / max(1.0, float(args.max_display_fps))

    try:
        while True:
            loop_started_at = time.perf_counter()
            frame_id, frame = camera_stream.get_latest_frame()
            camera_open, camera_error = camera_stream.get_camera_state()
            snapshot = inference_worker.get_snapshot()

            if frame is None:
                frame = build_placeholder_frame(
                    args.capture_width,
                    args.capture_height,
                    "RASPI1",
                    camera_error or "Waiting for camera",
                )

            now = time.perf_counter()
            delta = max(1e-6, now - last_display_time)
            instant_fps = 1.0 / delta
            display_fps = instant_fps if display_fps == 0.0 else ((display_fps * 0.9) + (instant_fps * 0.1))
            last_display_time = now

            annotated = draw_preview(
                frame=frame,
                snapshot=snapshot,
                camera_open=camera_open,
                camera_error=camera_error,
                display_fps=display_fps,
                inference_interval=inference_worker.get_inference_interval(),
                inference_mode=inference_worker.get_inference_mode(),
                image_size=current_image_size,
                profile_label=current_profile.label,
                max_display_fps=args.max_display_fps,
            )

            if web_server is not None:
                web_server.update_frame(annotated)

            if not headless:
                cv2.imshow(WINDOW_NAME, annotated)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = -1

            if key in (ord("q"), ord("Q"), 27):
                break

            if key in (ord("-"), ord("_")):
                new_threshold = inference_worker.adjust_threshold(-0.02)
                manual_threshold_override = True
                print(f"Threshold set to {new_threshold:.2f}")

            if key in (ord("+"), ord("=")):
                new_threshold = inference_worker.adjust_threshold(0.02)
                manual_threshold_override = True
                print(f"Threshold set to {new_threshold:.2f}")

            if key == ord("["):
                new_interval = inference_worker.adjust_inference_interval(-0.01)
                print(f"Inference gap set to {new_interval:.2f}s")

            if key == ord("]"):
                new_interval = inference_worker.adjust_inference_interval(0.01)
                print(f"Inference gap set to {new_interval:.2f}s")

            if key in (ord("r"), ord("R")):
                print("Reopening camera...")
                camera_stream.restart()

            if key in (ord("m"), ord("M")):
                next_mode = get_next_inference_mode(inference_worker.get_inference_mode())
                current_threshold = snapshot.threshold
                current_interval = inference_worker.get_inference_interval()

                try:
                    next_worker = build_inference_worker(
                        camera_stream=camera_stream,
                        model_bundle=current_model_bundle,
                        profile=current_profile,
                        class_names=current_class_names,
                        vehicle_class_ids=current_vehicle_class_ids,
                        threshold=current_threshold,
                        image_size=current_image_size,
                        inference_interval=current_interval,
                        inference_mode=next_mode,
                        iou_threshold=current_iou_threshold,
                        max_detections=current_max_detections,
                        warmup_model=False,
                    )
                except RuntimeError as error:
                    print(error)
                else:
                    inference_worker.stop()
                    inference_worker = next_worker
                    manual_mode_override = True
                    print(f"Inference mode set to {next_mode}")

            if key in (ord("c"), ord("C")):
                next_profile = get_next_profile(current_profile.model_path, available_profiles)

                if next_profile is None:
                    print("No switchable profiles were found in the project folder.")
                else:
                    next_model_bundle = load_model(next_profile.model_path)
                    next_threshold = snapshot.threshold
                    if not manual_threshold_override:
                        next_threshold = resolve_profile_threshold(args.threshold, next_model_bundle, next_profile)

                    current_interval = inference_worker.get_inference_interval()
                    next_image_size = resolve_profile_image_size(args.imgsz, next_profile)
                    next_class_names = resolve_profile_class_names(next_profile, next_model_bundle)
                    next_vehicle_class_ids = resolve_profile_vehicle_class_ids(next_profile, next_model_bundle)
                    next_mode = inference_worker.get_inference_mode()

                    if args.inference_mode == AUTO_INFERENCE_MODE and not manual_mode_override:
                        next_mode = resolve_profile_inference_mode(args.inference_mode, next_model_bundle, next_profile)

                    try:
                        next_worker = build_inference_worker(
                            camera_stream=camera_stream,
                            model_bundle=next_model_bundle,
                            profile=next_profile,
                            class_names=next_class_names,
                            vehicle_class_ids=next_vehicle_class_ids,
                            threshold=next_threshold,
                            image_size=next_image_size,
                            inference_interval=current_interval,
                            inference_mode=next_mode,
                            iou_threshold=current_iou_threshold,
                            max_detections=current_max_detections,
                            warmup_model=not args.skip_model_warmup,
                        )
                    except RuntimeError as error:
                        print(error)
                    else:
                        inference_worker.stop()
                        inference_worker = next_worker
                        current_threshold = next_threshold
                        current_model_bundle = next_model_bundle
                        current_profile = next_profile
                        current_image_size = next_image_size
                        current_class_names = next_class_names
                        current_vehicle_class_ids = next_vehicle_class_ids
                        save_selected_model(project_dir, current_profile.model_path)
                        print(f"Switched profile to {current_profile.label} ({next_mode} mode)")

            remaining = display_frame_period - (time.perf_counter() - loop_started_at)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        pass
    finally:
        if web_server is not None:
            web_server.stop()
        inference_worker.stop()
        camera_stream.stop()
        cv2.destroyAllWindows()

    return 0


def main() -> None:
    args = build_arg_parser().parse_args()
    raise SystemExit(run_local_preview(args))


if __name__ == "__main__":
    main()
