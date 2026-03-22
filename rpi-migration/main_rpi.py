import cv2
import threading
import time
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

cv2.setUseOptimized(True)

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

out = None
if WRITE_OUTPUT:
    out = cv2.VideoWriter(
        'output_video_rpi.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height),
    )

state_lock = threading.Lock()
last_boxes = []
latest_frame = None
latest_frame_idx = 0
running = True
inference_error = None


def inference_worker():
    global last_boxes, latest_frame, latest_frame_idx, running, inference_error
    processed_idx = -1

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

        with state_lock:
            last_boxes = current_boxes


infer_thread = threading.Thread(target=inference_worker, daemon=True)
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

cap.release()
running = False
infer_thread.join(timeout=2.0)
if out is not None:
    out.release()
cv2.destroyAllWindows()
if out is not None:
    print('Saved output: output_video_rpi.mp4')
if inference_error:
    raise RuntimeError(f'Inference worker failed: {inference_error}')
