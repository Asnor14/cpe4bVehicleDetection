#!/usr/bin/env python3
"""Webcam-only TensorFlow Lite vehicle detection."""

import argparse
import importlib.util
import os
import subprocess
import time
from threading import Thread

import cv2
import numpy as np


DEFAULT_LABELS = [
    "traffic sign",
    "traffic light",
    "car",
    "rider",
    "motor",
    "person",
    "bus",
    "truck",
    "bike",
    "train",
]

VEHICLE_CLASSES = {
    "car",
    "bus",
    "truck",
    "motor",
    "bike",
    "train",
    "motorbike",
    "motorcycle",
    "bicycle",
}


def parse_normalized_roi(roi_text):
    try:
        values = [float(value.strip()) for value in roi_text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "ROI must have 4 comma-separated numeric values: x1,y1,x2,y2"
        ) from exc

    if len(values) != 4:
        raise argparse.ArgumentTypeError("ROI must have 4 comma-separated values: x1,y1,x2,y2")

    left, top, right, bottom = values
    if not all(0.0 <= value <= 1.0 for value in values):
        raise argparse.ArgumentTypeError("ROI values must be between 0.0 and 1.0")
    if left >= right or top >= bottom:
        raise argparse.ArgumentTypeError("ROI must satisfy x1 < x2 and y1 < y2")

    return left, top, right, bottom


def denormalize_roi(roi, image_width, image_height):
    left, top, right, bottom = roi
    return (
        int(left * image_width),
        int(top * image_height),
        int(right * image_width),
        int(bottom * image_height),
    )


def point_in_roi(x, y, roi):
    left, top, right, bottom = roi
    return left <= x <= right and top <= y <= bottom


def load_labels(labels_path):
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]

    if labels and labels[0] == "???":
        labels = labels[1:]

    if labels != DEFAULT_LABELS:
        if len(labels) != len(DEFAULT_LABELS):
            print(
                "Warning: label map length does not match the bundled model. "
                "Falling back to the built-in class order."
            )
            return DEFAULT_LABELS.copy()

        expected_vehicle_labels = {"car", "rider", "motor", "person", "bus", "truck", "bike"}
        if not expected_vehicle_labels.issubset(set(labels)):
            print(
                "Warning: label map contents look inconsistent with the bundled model. "
                "Falling back to the built-in class order."
            )
            return DEFAULT_LABELS.copy()

    return labels


class VideoStream:
    """Camera object that reads video frames on a background thread."""

    def __init__(self, camera_id=0, resolution=(1280, 720)):
        self.stream = cv2.VideoCapture(camera_id)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

        self.grabbed, self.frame = self.stream.read()
        if not self.grabbed:
            raise RuntimeError("Could not read frame from webcam.")

        self.stopped = False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            self.grabbed, self.frame = self.stream.read()
            if not self.grabbed:
                self.stop()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()


class CsiVideoStream:
    """CSI camera reader backed by Picamera2/libcamera."""

    def __init__(self, resolution=(1280, 720)):
        try:
            from picamera2 import Picamera2  # pylint: disable=import-error
        except ImportError as exc:
            raise RuntimeError(
                "CSI mode requires Picamera2. Install with: sudo apt install -y python3-picamera2"
            ) from exc

        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": resolution, "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()

    def start(self):
        return self

    def read(self):
        frame_rgb = self.picam2.capture_array()
        if frame_rgb is None:
            return None
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    def stop(self):
        self.picam2.stop()


def run_csi_test(timeout_seconds):
    """Run rpicam-hello for quick CSI camera validation."""
    print(f"Running CSI test: rpicam-hello -t {timeout_seconds * 1000}")
    subprocess.run(
        ["rpicam-hello", "-t", str(timeout_seconds * 1000)],
        check=True,
    )


def build_camera_stream(args, im_w, im_h):
    if args.camera_source == "usb":
        return VideoStream(camera_id=args.camera_id, resolution=(im_w, im_h)).start()
    return CsiVideoStream(resolution=(im_w, im_h)).start()


def build_interpreter(model_path, use_tpu):
    pkg = importlib.util.find_spec("tensorflow")
    if pkg is None:
        from tflite_runtime.interpreter import Interpreter
        if use_tpu:
            from tflite_runtime.interpreter import load_delegate
    else:
        from tensorflow.lite.python.interpreter import Interpreter
        if use_tpu:
            from tensorflow.lite.python.interpreter import load_delegate

    if use_tpu:
        return Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate("libedgetpu.so.1.0")],
        )
    return Interpreter(model_path=model_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modeldir", required=True, help="Folder containing .tflite and labels")
    parser.add_argument("--graph", default="detect.tflite", help="Name of the .tflite model")
    parser.add_argument("--labels", default="labelmap.txt", help="Name of the label map file")
    parser.add_argument("--threshold", default=0.5, type=float, help="Minimum score threshold")
    parser.add_argument("--resolution", default="1280x720", help="Webcam resolution, for example 1280x720")
    parser.add_argument(
        "--camera-source",
        default="usb",
        choices=["usb", "csi"],
        help="Camera backend to use: usb (OpenCV device) or csi (Picamera2/libcamera)",
    )
    parser.add_argument("--camera-id", default=0, type=int, help="OpenCV camera index")
    parser.add_argument(
        "--test-csi",
        action="store_true",
        help="Run 'rpicam-hello' test and exit",
    )
    parser.add_argument(
        "--test-csi-seconds",
        default=8,
        type=int,
        help="Duration for --test-csi",
    )
    parser.add_argument(
        "--roi",
        default=parse_normalized_roi("0.35,0.55,0.65,0.95"),
        type=parse_normalized_roi,
        help="Normalized ROI rectangle as x1,y1,x2,y2",
    )
    parser.add_argument(
        "--hold-seconds",
        default=5.0,
        type=float,
        help="How long to keep the vehicle-detected sign active after the last hit",
    )
    parser.add_argument("--edgetpu", action="store_true", help="Use Coral Edge TPU")
    args = parser.parse_args()

    if args.test_csi:
        run_csi_test(args.test_csi_seconds)
        return

    res_w, res_h = args.resolution.split("x")
    im_w, im_h = int(res_w), int(res_h)
    roi_pixels = denormalize_roi(args.roi, im_w, im_h)

    model_path = os.path.join(os.getcwd(), args.modeldir, args.graph)
    labels_path = os.path.join(os.getcwd(), args.modeldir, args.labels)

    labels = load_labels(labels_path)

    interpreter = build_interpreter(model_path, args.edgetpu)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]["shape"][1]
    width = input_details[0]["shape"][2]
    floating_model = input_details[0]["dtype"] == np.float32

    input_mean = 127.5
    input_std = 127.5

    videostream = build_camera_stream(args, im_w, im_h)
    time.sleep(1)

    frame_rate_calc = 0.0
    freq = cv2.getTickFrequency()
    vehicle_hold_until = 0.0

    print(
        f"Vehicle detection started using '{args.camera_source}' camera. "
        "Press 'q' to quit."
    )

    try:
        while True:
            t1 = cv2.getTickCount()
            now = time.monotonic()

            frame = videostream.read()
            if frame is None:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (width, height))
            input_data = np.expand_dims(frame_resized, axis=0)

            if floating_model:
                input_data = (np.float32(input_data) - input_mean) / input_std

            interpreter.set_tensor(input_details[0]["index"], input_data)
            interpreter.invoke()

            boxes = interpreter.get_tensor(output_details[0]["index"])[0]
            classes = interpreter.get_tensor(output_details[1]["index"])[0]
            scores = interpreter.get_tensor(output_details[2]["index"])[0]

            for i, score in enumerate(scores):
                if score < args.threshold or score > 1.0:
                    continue

                class_idx = int(classes[i])
                if class_idx >= len(labels):
                    continue

                object_name = labels[class_idx].lower()
                if object_name not in VEHICLE_CLASSES:
                    continue

                ymin = int(max(1, boxes[i][0] * im_h))
                xmin = int(max(1, boxes[i][1] * im_w))
                ymax = int(min(im_h, boxes[i][2] * im_h))
                xmax = int(min(im_w, boxes[i][3] * im_w))
                center_x = (xmin + xmax) // 2
                center_y = (ymin + ymax) // 2
                in_zone = point_in_roi(center_x, center_y, roi_pixels)

                if in_zone:
                    vehicle_hold_until = now + args.hold_seconds

                box_color = (10, 255, 0) if in_zone else (0, 180, 255)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), box_color, 2)
                cv2.circle(frame, (center_x, center_y), 4, box_color, cv2.FILLED)
                label = f"{object_name}: {int(score * 100)}%"
                label_size, base_line = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                label_ymin = max(ymin, label_size[1] + 10)
                cv2.rectangle(
                    frame,
                    (xmin, label_ymin - label_size[1] - 10),
                    (xmin + label_size[0], label_ymin + base_line - 10),
                    (255, 255, 255),
                    cv2.FILLED,
                )
                cv2.putText(
                    frame,
                    label,
                    (xmin, label_ymin - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                )

            display_now = time.monotonic()
            vehicle_detected = display_now <= vehicle_hold_until
            t2 = cv2.getTickCount()
            time_per_frame = (t2 - t1) / freq
            frame_rate_calc = 1 / time_per_frame if time_per_frame > 0 else 0

            roi_color = (0, 255, 0) if vehicle_detected else (0, 120, 0)
            cv2.rectangle(
                frame,
                (roi_pixels[0], roi_pixels[1]),
                (roi_pixels[2], roi_pixels[3]),
                roi_color,
                2,
            )
            cv2.putText(
                frame,
                "Detection zone",
                (roi_pixels[0], max(roi_pixels[1] - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                roi_color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"FPS: {frame_rate_calc:.2f}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            if vehicle_detected:
                cv2.putText(
                    frame,
                    "Vehicle detected",
                    (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                remaining = max(0.0, vehicle_hold_until - display_now)
                cv2.putText(
                    frame,
                    f"Hold: {remaining:.1f}s",
                    (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imshow("Vehicle detector", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        videostream.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
