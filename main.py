import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from ultralytics import YOLO


# Web server settings.
HOST = "0.0.0.0"
PORT = 5000
DEBUG = False

# Model and camera settings.
MODEL_NAME = "best.pt"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
IMAGE_SIZE = 640
JPEG_QUALITY = 80
DEFAULT_CONFIDENCE = 0.85
WARNING_SECONDS = 5.0

# Two device panels are shown on the page.
# RASPI1 is enabled by default for your current PC webcam.
# RASPI2 starts as disabled, so the panel shows an offline placeholder until you enable it later.
DEVICE_CONFIGS = [
    {
        "id": "raspi1",
        "name": "RASPI1",
        "camera_index": 0,
        "enabled": True,
        "confidence_threshold": DEFAULT_CONFIDENCE,
    },
    {
        "id": "raspi2",
        "name": "RASPI2",
        "camera_index": 1,
        "enabled": False,
        "confidence_threshold": DEFAULT_CONFIDENCE,
    },
]

# Common vehicle words used to filter detections when class names exist.
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
    "van",
}


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


def open_camera(camera_index: int):
    """Try a few camera backends for better cross-platform webcam support."""
    if os.name == "nt":
        attempts = [
            getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
            cv2.CAP_ANY,
            getattr(cv2, "CAP_MSMF", cv2.CAP_ANY),
        ]
    else:
        attempts = [getattr(cv2, "CAP_V4L2", cv2.CAP_ANY), cv2.CAP_ANY]

    for backend in attempts:
        cap = cv2.VideoCapture(camera_index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

            if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            return cap

        cap.release()

    return None


class DeviceDetector:
    """Own one camera stream and keep its latest detection state."""

    def __init__(
        self,
        config: dict,
        model,
        model_lock: threading.Lock,
        class_names: dict[int, str],
        vehicle_class_ids: set[int],
        startup_error: str | None = None,
    ) -> None:
        self.id = config["id"]
        self.name = config["name"]
        self.camera_index = int(config["camera_index"])
        self.enabled = bool(config["enabled"])
        self.threshold = clamp_threshold(config["confidence_threshold"])
        self.model = model
        self.model_lock = model_lock
        self.class_names = class_names
        self.vehicle_class_ids = vehicle_class_ids
        self.startup_error = startup_error

        self.cap = None
        self.thread = None
        self.stop_event = threading.Event()
        self.frame_lock = threading.Lock()
        self.state_lock = threading.Lock()

        self.latest_jpeg = self._build_placeholder_frame("Waiting for camera")
        self.status_message = "Starting"
        self.last_error = ""
        self.camera_open = False
        self.last_detection_time = 0.0
        self.last_detection_confidence = 0.0

        if not self.enabled:
            self.status_message = "Offline"
            self.last_error = "This panel is disabled for now."
            self.latest_jpeg = self._build_placeholder_frame("Device disabled")
        elif self.startup_error:
            self.status_message = "Model error"
            self.last_error = self.startup_error
            self.latest_jpeg = self._build_placeholder_frame("Model missing")

    def start(self) -> None:
        """Start the background camera thread for an enabled device."""
        if not self.enabled or self.startup_error or self.thread is not None:
            return

        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def restart_camera(self) -> None:
        """Try reopening the webcam when the user presses Refresh."""
        if not self.enabled or self.startup_error:
            return

        with self.state_lock:
            self.status_message = "Reconnecting"
            self.last_error = ""

        self._release_camera()

    def update_threshold(self, new_threshold: float) -> float:
        """Save a new confidence threshold from the web page."""
        with self.state_lock:
            self.threshold = clamp_threshold(new_threshold)
            return self.threshold

    def stream_frames(self):
        """Yield the latest JPEG frame as an MJPEG stream for the browser."""
        while True:
            with self.frame_lock:
                frame_bytes = self.latest_jpeg

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

            time.sleep(0.05)

    def get_status(self) -> dict:
        """Return the current UI state for this device panel."""
        with self.state_lock:
            seconds_remaining = max(0.0, WARNING_SECONDS - (time.time() - self.last_detection_time))
            warning_active = seconds_remaining > 0.0

            return {
                "id": self.id,
                "name": self.name,
                "enabled": self.enabled,
                "camera_open": self.camera_open,
                "threshold": round(self.threshold, 2),
                "warning_active": warning_active,
                "countdown_seconds": round(seconds_remaining, 1),
                "last_detection_confidence": round(self.last_detection_confidence, 2),
                "status_message": self.status_message,
                "last_error": self.last_error,
            }

    def _run_loop(self) -> None:
        """Read frames, run YOLO, and store the newest JPEG."""
        while not self.stop_event.is_set():
            if self.cap is None:
                self.cap = open_camera(self.camera_index)

                if self.cap is None:
                    self._set_camera_error(
                        f"Could not open webcam at camera index {self.camera_index}."
                    )
                    time.sleep(2.0)
                    continue

                with self.state_lock:
                    self.camera_open = True
                    self.status_message = "Live"
                    self.last_error = ""

            success, frame = self.cap.read()

            if not success or frame is None:
                self._set_camera_error("Failed to read a frame from the webcam.")
                self._release_camera()
                time.sleep(1.0)
                continue

            annotated_frame, detected_vehicle = self._process_frame(frame)
            encoded = self._encode_frame(annotated_frame)

            with self.frame_lock:
                self.latest_jpeg = encoded

            if detected_vehicle:
                with self.state_lock:
                    self.last_detection_time = time.time()

    def _process_frame(self, frame):
        """Run YOLO on one frame and draw vehicle boxes on the image."""
        detected_vehicle = False
        best_confidence = 0.0
        annotated_frame = frame.copy()

        if self.model is None:
            with self.state_lock:
                self.status_message = "Model error"
                self.last_error = self.startup_error or "Model is not available."
            return self._draw_footer(annotated_frame, "Model error"), False

        with self.state_lock:
            threshold = self.threshold

        with self.model_lock:
            results = self.model.predict(
                source=frame,
                conf=threshold,
                imgsz=IMAGE_SIZE,
                device="cpu",
                verbose=False,
            )

        boxes = results[0].boxes

        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                if self.vehicle_class_ids and class_id not in self.vehicle_class_ids:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                label = f"Vehicle {confidence:.2f}"

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (48, 219, 91), 2)

                label_y = y1 - 10 if y1 > 24 else y1 + 24
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (48, 219, 91),
                    2,
                    cv2.LINE_AA,
                )

                if confidence >= best_confidence:
                    best_confidence = confidence
                    detected_vehicle = True

        with self.state_lock:
            if detected_vehicle:
                self.last_detection_confidence = best_confidence
                self.status_message = "Vehicle detected"
                self.last_error = ""
            else:
                self.status_message = "Watching"

        footer_text = f"Vehicle {best_confidence:.2f}" if detected_vehicle else "No vehicle detected"
        annotated_frame = self._draw_footer(annotated_frame, footer_text)
        return annotated_frame, detected_vehicle

    def _draw_footer(self, frame, text: str):
        """Draw a simple footer bar for the preview frame."""
        cv2.rectangle(
            frame,
            (0, FRAME_HEIGHT - 34),
            (FRAME_WIDTH, FRAME_HEIGHT),
            (18, 18, 18),
            -1,
        )
        cv2.putText(
            frame,
            text,
            (12, FRAME_HEIGHT - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return frame

    def _build_placeholder_frame(self, text: str) -> bytes:
        """Create a simple JPEG placeholder when a camera is offline."""
        image = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), (236, 236, 236), dtype=np.uint8)
        cv2.rectangle(image, (8, 8), (FRAME_WIDTH - 8, FRAME_HEIGHT - 8), (32, 32, 32), 2)
        cv2.putText(
            image,
            self.name,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (32, 32, 32),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            text,
            (20, FRAME_HEIGHT // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (88, 88, 88),
            2,
            cv2.LINE_AA,
        )
        return self._encode_frame(image)

    def _encode_frame(self, frame) -> bytes:
        """Convert an OpenCV image into JPEG bytes for the web browser."""
        success, buffer = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
        )

        if not success:
            fallback = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 220, dtype=np.uint8)
            cv2.putText(
                fallback,
                "Frame encode failed",
                (20, FRAME_HEIGHT // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (20, 20, 20),
                2,
                cv2.LINE_AA,
            )
            success, buffer = cv2.imencode(".jpg", fallback)

        return buffer.tobytes()

    def _set_camera_error(self, message: str) -> None:
        """Save a readable camera error and swap in a placeholder frame."""
        with self.state_lock:
            self.camera_open = False
            self.status_message = "Camera error"
            self.last_error = message

        with self.frame_lock:
            self.latest_jpeg = self._build_placeholder_frame(message)

    def _release_camera(self) -> None:
        """Release the webcam cleanly."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        with self.state_lock:
            self.camera_open = False


def create_app():
    """Create the Flask app and detector objects."""
    app = Flask(__name__)
    project_dir = Path(__file__).resolve().parent
    model_path = project_dir / MODEL_NAME

    model = None
    model_error = None
    class_names = {}
    vehicle_class_ids = set()
    model_lock = threading.Lock()

    if not model_path.exists():
        model_error = f"Error: '{MODEL_NAME}' was not found in the same folder as main.py."
        print(model_error)
    else:
        try:
            model = YOLO(str(model_path))
            class_names = normalize_names(getattr(model, "names", None))
            vehicle_class_ids = get_vehicle_class_ids(class_names)

            if vehicle_class_ids:
                print("Vehicle class filter enabled.")
            else:
                print("Note: no vehicle-like class names were found, so all detections will be shown.")
        except Exception as error:
            model_error = f"Error: failed to load model '{MODEL_NAME}'. Details: {error}"
            print(model_error)

    devices = {
        config["id"]: DeviceDetector(
            config=config,
            model=model,
            model_lock=model_lock,
            class_names=class_names,
            vehicle_class_ids=vehicle_class_ids,
            startup_error=model_error,
        )
        for config in DEVICE_CONFIGS
    }

    for device in devices.values():
        device.start()

    @app.route("/")
    def index():
        return render_template(
            "index.html",
            devices=[device.get_status() for device in devices.values()],
            warning_seconds=WARNING_SECONDS,
        )

    @app.route("/video_feed/<device_id>")
    def video_feed(device_id: str):
        device = devices.get(device_id)

        if device is None:
            return Response("Unknown device", status=404)

        return Response(
            device.stream_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/api/status")
    def api_status():
        return jsonify(
            {
                "warning_seconds": WARNING_SECONDS,
                "devices": {
                    device_id: device.get_status()
                    for device_id, device in devices.items()
                },
            }
        )

    @app.route("/api/device/<device_id>/threshold", methods=["POST"])
    def update_threshold(device_id: str):
        device = devices.get(device_id)

        if device is None:
            return jsonify({"ok": False, "message": "Unknown device"}), 404

        request_json = request.get_json(silent=True) or {}
        raw_value = request.form.get("threshold") or request_json.get("threshold")

        try:
            new_threshold = clamp_threshold(float(raw_value))
        except (TypeError, ValueError):
            return jsonify({"ok": False, "message": "Enter a number from 0.01 to 1.00."}), 400

        saved_threshold = device.update_threshold(new_threshold)
        return jsonify(
            {
                "ok": True,
                "device_id": device_id,
                "threshold": saved_threshold,
            }
        )

    @app.route("/api/device/<device_id>/refresh", methods=["POST"])
    def refresh_device(device_id: str):
        device = devices.get(device_id)

        if device is None:
            return jsonify({"ok": False, "message": "Unknown device"}), 404

        device.restart_camera()
        return jsonify({"ok": True, "device_id": device_id})

    return app


app = create_app()


def main() -> None:
    print(f"Open http://localhost:{PORT} on this computer.")
    print(f"Open http://<your-pc-or-pi-ip>:{PORT} from another device on the same network.")
    app.run(host=HOST, port=PORT, debug=DEBUG, threaded=True)


if __name__ == "__main__":
    main()
