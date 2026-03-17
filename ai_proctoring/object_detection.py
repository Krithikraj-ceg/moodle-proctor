"""
Object Detection Module
Detects forbidden exam objects using YOLOv8:
  - Books / notebooks
  - Earphones / headphones
  - Calculators
  - Laptops / tablets (secondary screens)
  - Remotes (suspicious)

Uses COCO class IDs for common objects + phone (already in phone_detection).
"""

import cv2
import time

from violation_logger import ViolationLogger, ViolationType
from utils import capture_screenshot


# COCO class IDs for forbidden items
FORBIDDEN_OBJECTS = {
    67:  ("Cell Phone",   0.45),
    73:  ("Book",         0.40),
    76:  ("Scissors",     0.45),
    63:  ("Laptop",       0.45),
    62:  ("TV/Monitor",   0.45),
    65:  ("Remote",       0.45),
}

EARPHONE_KEYWORDS  = ["earphone", "headphone", "airpod"]   # YOLO custom if available
EVENT_COOLDOWN_SEC = 5.0
BBOX_COLORS = {
    "Cell Phone":  (0,   50,  255),
    "Book":        (0,   165, 255),
    "Scissors":    (0,   0,   200),
    "Laptop":      (255, 50,  50 ),
    "TV/Monitor":  (255, 100, 0  ),
    "Remote":      (200, 0,   200),
}


class ObjectDetector:
    """Detect forbidden exam objects via YOLOv8."""

    def __init__(self, logger: ViolationLogger):
        self.logger = logger
        self._model = None
        self._last_event: dict = {}   # class_id → last event timestamp
        self._load_model()

    def process(self, frame_bgr, annotated_bgr, frame_index: int) -> dict:
        if self._model is None:
            return {"detections": []}

        results = self._model.predict(
            source  = frame_bgr,
            conf    = 0.35,
            classes = list(FORBIDDEN_OBJECTS.keys()),
            verbose = False,
            imgsz   = 320,
        )

        found = []
        if results and len(results[0].boxes):
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                if cls_id not in FORBIDDEN_OBJECTS:
                    continue
                name, min_conf = FORBIDDEN_OBJECTS[cls_id]
                if conf < min_conf:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                found.append({"class_id": cls_id, "name": name,
                              "confidence": conf, "bbox": [x1, y1, x2, y2]})

        for item in found:
            self._draw_box(annotated_bgr, item)
            self._fire_violation(frame_bgr, frame_index, item)

        return {"detections": found, "count": len(found)}

    def _fire_violation(self, frame, frame_index, item):
        now     = time.time()
        cls_id  = item["class_id"]
        if now - self._last_event.get(cls_id, 0) < EVENT_COOLDOWN_SEC:
            return
        path = capture_screenshot(frame, frame_index, f"object_{item['name'].lower().replace('/', '_')}")
        self.logger.log(
            violation_type  = ViolationType.FORBIDDEN_OBJECT,
            confidence      = round(item["confidence"], 4),
            screenshot_path = path,
            extra           = {"object": item["name"], "bbox": item["bbox"]},
        )
        self._last_event[cls_id] = now

    def _draw_box(self, annotated, item):
        x1, y1, x2, y2 = item["bbox"]
        name  = item["name"]
        conf  = item["confidence"]
        color = BBOX_COLORS.get(name, (0, 0, 200))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated, (x1, y1-th-12), (x1+tw+6, y1), color, -1)
        cv2.putText(annotated, label, (x1+3, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

    def _load_model(self):
        try:
            from ultralytics import YOLO
            import numpy as np
            self._model = YOLO("yolov8n.pt")
            dummy = np.zeros((320, 320, 3), dtype="uint8")
            self._model.predict(source=dummy, conf=0.5, verbose=False, imgsz=320)
            print("[ObjectDetector] Loaded YOLOv8 for forbidden object detection.")
        except Exception as exc:
            print(f"[ObjectDetector] WARNING: {exc}")
