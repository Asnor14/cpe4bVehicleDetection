import cv2
import threading
import time

import RPi.GPIO as GPIO
from ultralytics import YOLO

# Raspberry Pi tuned defaults
MODEL_PATH = 'yolo11n.pt'
CAM_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 360
TARGET_FPS = 30
IMG_SIZE = 512
FRAME_SKIP = 1  # process every frame for stronger small-vehicle detection
DETECT_CONF = 0.15
WRITE_OUTPUT = False  # disable recording for smoother live display
GENERIC_LABEL = 'Vehicle'
VEHICLE_CLASSES = [1, 2, 3, 5, 6, 7]

RELAY_ON_PIN = 17
RELAY_OFF_PIN = 27
RELAY_ACTIVE_SECONDS = 5.0
RELAY_TRIGGER_PULSE_SECONDS = 0.2

cv2.setUseOptimized(True)

state_lock = threading.Lock()
relay_lock = threading.Lock()
last_boxes = []
latest_frame = None
latest_frame_idx = 0
running = True
inference_error = None
relay_sequence_active = False
relay_thread = None
relay_stop_event = threading.Event()


def set_relay_idle() -> None:
    GPIO.output(RELAY_ON_PIN, GPIO.HIGH)
    GPIO.output(RELAY_OFF_PIN, GPIO.HIGH)


def initialize_gpio() -> None:
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(RELAY_ON_PIN, GPIO.OUT, initial=GPIO.HIGH)
    GPIO.setup(RELAY_OFF_PIN, GPIO.OUT, initial=GPIO.HIGH)
    set_relay_idle()


def relay_on() -> None:
    with relay_lock:
        GPIO.output(RELAY_OFF_PIN, GPIO.HIGH)
        GPIO.output(RELAY_ON_PIN, GPIO.LOW)

    time.sleep(RELAY_TRIGGER_PULSE_SECONDS)

    with relay_lock:
        GPIO.output(RELAY_ON_PIN, GPIO.HIGH)


def relay_off() -> None:
    with relay_lock:
        GPIO.output(RELAY_ON_PIN, GPIO.HIGH)
        GPIO.output(RELAY_OFF_PIN, GPIO.LOW)

    time.sleep(RELAY_TRIGGER_PULSE_SECONDS)

    with relay_lock:
        GPIO.output(RELAY_OFF_PIN, GPIO.HIGH)


def relay_sequence_worker() -> None:
    global relay_sequence_active

    try:
        if relay_stop_event.is_set():
            return

        relay_on()

        if relay_stop_event.wait(RELAY_ACTIVE_SECONDS):
            return

        relay_off()
    finally:
        with relay_lock:
            set_relay_idle()
            relay_sequence_active = False


def trigger_relay_sequence() -> None:
    global relay_sequence_active, relay_thread

    with relay_lock:
        if relay_sequence_active or relay_stop_event.is_set():
            return

        relay_sequence_active = True
        relay_thread = threading.Thread(target=relay_sequence_worker, daemon=True)
        relay_thread.start()


def cleanup_gpio() -> None:
    with relay_lock:
        set_relay_idle()
        GPIO.cleanup((RELAY_ON_PIN, RELAY_OFF_PIN))


def inference_worker(model: YOLO) -> None:
    global last_boxes, latest_frame, latest_frame_idx, running, inference_error
    processed_idx = -1
    relay_trigger_armed = True

    while running:
        with state_lock:
            frame_idx = latest_frame_idx
            frame_for_infer = (
                latest_frame.copy()
                if latest_frame is not None and frame_idx != processed_idx
                else None
            )

        if frame_for_infer is None:
            time.sleep(0.002)
            continue

        processed_idx = frame_idx
        if frame_idx % FRAME_SKIP != 0:
            continue

        try:
            results = model.predict(
                source=frame_for_infer,
                classes=VEHICLE_CLASSES,
                imgsz=IMG_SIZE,
                conf=DETECT_CONF,
                device='cpu',
                verbose=False,
            )
        except Exception as exc:
            inference_error = str(exc)
            running = False
            break

        current_boxes = []
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu()
            confidences = results[0].boxes.conf.cpu().tolist()

            for box, confidence in zip(boxes, confidences):
                x1, y1, x2, y2 = map(int, box)
                current_boxes.append((x1, y1, x2, y2, float(confidence)))

        # Trigger once per detection episode so a single vehicle does not pulse every frame.
        if current_boxes:
            if relay_trigger_armed:
                trigger_relay_sequence()
                relay_trigger_armed = False
        else:
            relay_trigger_armed = True

        with state_lock:
            last_boxes = current_boxes


def main() -> None:
    global latest_frame, latest_frame_idx, running

    cap = None
    out = None
    infer_thread = None
    gpio_ready = False

    try:
        initialize_gpio()
        gpio_ready = True

        model = YOLO(MODEL_PATH)

        cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f'Could not open camera index {CAM_INDEX}.')

        # Request MJPG from webcam to reduce USB and decode overhead.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or FRAME_WIDTH
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or FRAME_HEIGHT
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or TARGET_FPS

        if WRITE_OUTPUT:
            out = cv2.VideoWriter(
                'output_video_rpi.mp4',
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (frame_width, frame_height),
            )

        infer_thread = threading.Thread(target=inference_worker, args=(model,), daemon=True)
        infer_thread.start()

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            with state_lock:
                latest_frame = frame
                latest_frame_idx = frame_idx
                boxes_snapshot = list(last_boxes)

            for x1, y1, x2, y2, confidence in boxes_snapshot:
                cv2.putText(
                    frame,
                    f'{GENERIC_LABEL} {confidence:.2f}',
                    (x1, max(15, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                )
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if out is not None:
                out.write(frame)
            cv2.imshow('YOLO Vehicle Detection (RPi)', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if inference_error:
                break
    finally:
        running = False
        relay_stop_event.set()

        if infer_thread is not None:
            infer_thread.join(timeout=2.0)
        if relay_thread is not None:
            relay_thread.join(timeout=1.0)
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()

        cv2.destroyAllWindows()

        if gpio_ready:
            cleanup_gpio()

    if out is not None:
        print('Saved output: output_video_rpi.mp4')
    if inference_error:
        raise RuntimeError(f'Inference worker failed: {inference_error}')


if __name__ == '__main__':
    main()
