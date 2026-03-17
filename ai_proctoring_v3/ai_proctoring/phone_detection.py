"""
Phone Detection Module
Uses YOLOv8 (ultralytics) to detect mobile phones (COCO class 67).
Falls back gracefully if the model weights aren't downloaded yet.

On first run the weights are auto-downloaded by ultralytics (~6 MB for nano).
"""

import cv2
import time

from violation_logger import ViolationLogger, ViolationType
from utils import capture_screenshot

# ─── Tuning ───────────────────────────────────────────────────────────────────
YOLO_MODEL_NAME    = "yolov8n.pt"     # Nano = fastest; swap for yolov8s/m for accuracy
PHONE_CONF_THRESH  = 0.45             # Minimum confidence to flag
PHONE_CLASS_ID     = 67               # COCO class 67 = "cell phone"
EVENT_COOLDOWN_SEC = 4.0              # Min seconds between repeated phone events
BBOX_COLOR         = (255, 60, 20)    # Blue-ish red (BGR)


class PhoneDetector:
    """YOLO-based mobile phone detector."""

    def __init__(self, logger: ViolationLogger):
        self.logger = logger
        self._model = None
        self._last_event_time = 0.0
        self._load_model()

    # ── Public ────────────────────────────────────────────────────────────────

    def process(self, frame_bgr, annotated_bgr, frame_index: int) -> dict:
        if self._model is None:
            return {"phone_detected": False, "detections": []}

        # Inference (returns list of Results objects)
        results = self._model.predict(
            source   = frame_bgr,
            conf     = PHONE_CONF_THRESH,
            classes  = [PHONE_CLASS_ID],
            verbose  = False,
            imgsz    = 320,            # Smaller → faster inference
        )

        phone_boxes = []
        if results and len(results[0].boxes):
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                if cls_id == PHONE_CLASS_ID:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    phone_boxes.append({"bbox": [x1, y1, x2, y2], "confidence": conf})

        if phone_boxes:
            self._draw_boxes(annotated_bgr, phone_boxes)
            self._fire_violation(frame_bgr, frame_index, phone_boxes)

        return {
            "phone_detected": len(phone_boxes) > 0,
            "detections":      phone_boxes,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_model(self):
        try:
            from ultralytics import YOLO
            self._model = YOLO(YOLO_MODEL_NAME)
            # Warm-up pass with a blank frame to avoid latency on first real frame
            import numpy as np
            dummy = np.zeros((320, 320, 3), dtype="uint8")
            self._model.predict(source=dummy, conf=0.5, verbose=False, imgsz=320)
            print(f"[PhoneDetector] YOLO model '{YOLO_MODEL_NAME}' loaded & warmed up.")
        except ImportError:
            print("[PhoneDetector] WARNING: ultralytics not installed. "
                  "Phone detection disabled. Run: pip install ultralytics")
        except Exception as exc:
            print(f"[PhoneDetector] WARNING: Failed to load YOLO model — {exc}. "
                  "Phone detection disabled.")

    def _fire_violation(self, frame, frame_index, boxes):
        now = time.time()
        if now - self._last_event_time < EVENT_COOLDOWN_SEC:
            return

        best       = max(boxes, key=lambda b: b["confidence"])
        path       = capture_screenshot(frame, frame_index, "phone")
        self.logger.log(
            violation_type  = ViolationType.PHONE_DETECTED,
            confidence      = round(best["confidence"], 4),
            screenshot_path = path,
            extra           = {
                "phones_count": len(boxes),
                "bbox":         best["bbox"],
            },
        )
        self._last_event_time = now

    def _draw_boxes(self, annotated, boxes):
        for item in boxes:
            x1, y1, x2, y2 = item["bbox"]
            conf            = item["confidence"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), BBOX_COLOR, 2)
            label = f"PHONE  {conf:.2f}"
            # Background pill for label readability
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 12), (x1 + tw + 6, y1), BBOX_COLOR, -1)
            cv2.putText(
                annotated, label, (x1 + 3, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 1, cv2.LINE_AA,
            )
            cv2.putText(
                annotated, "! PHONE DETECTED !",
                (20, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                BBOX_COLOR, 2, cv2.LINE_AA,
            )
