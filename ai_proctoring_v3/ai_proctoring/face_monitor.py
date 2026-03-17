"""
Face Monitor Module
Handles:
  - Face presence detection (no-face violation after 5 s)
  - Multiple-face detection (immediate violation)

Uses MediaPipe Tasks FaceDetector API (compatible with mediapipe >= 0.10).
"""

import cv2
import time
import urllib.request
import os
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from violation_logger import ViolationLogger, ViolationType
from utils import capture_screenshot


# ─── Tuning Constants ─────────────────────────────────────────────────────────
NO_FACE_TIMEOUT_SEC  = 5.0
MIN_DETECTION_CONF   = 0.6
BBOX_COLOR_OK        = (0, 220, 80)
BBOX_COLOR_WARN      = (0, 165, 255)
BBOX_COLOR_VIOL      = (0, 0, 220)

MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
MODEL_PATH = "blaze_face_short_range.tflite"


def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[FaceMonitor] Downloading face detection model …")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"[FaceMonitor] Model saved to '{MODEL_PATH}'.")


class FaceMonitor:
    """Detect face count per frame and emit violations when rules are broken."""

    def __init__(self, logger: ViolationLogger):
        self.logger = logger
        _ensure_model()

        base_options    = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options         = mp_vision.FaceDetectorOptions(
            base_options            = base_options,
            min_detection_confidence= MIN_DETECTION_CONF,
        )
        self.detector = mp_vision.FaceDetector.create_from_options(options)

        # State tracking
        self._no_face_start         = None
        self._no_face_violated      = False
        self._multi_face_last_event = 0.0

        print("[FaceMonitor] Initialised (MediaPipe Tasks FaceDetector).")

    # ── Public ────────────────────────────────────────────────────────────────

    def process(self, frame_bgr, annotated_bgr, frame_index: int) -> dict:
        rgb_frame  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result     = self.detector.detect(mp_image)
        detections = result.detections or []
        face_count = len(detections)

        self._draw_detections(annotated_bgr, detections, frame_bgr.shape)
        self._check_no_face(face_count, frame_bgr, annotated_bgr, frame_index)
        self._check_multiple_faces(face_count, frame_bgr, annotated_bgr, frame_index, detections)

        return {
            "face_count":    face_count,
            "violation":     face_count == 0 or face_count > 1,
            "violation_type": (
                ViolationType.NO_FACE    if face_count == 0 else
                ViolationType.MULTI_FACE if face_count > 1  else
                None
            ),
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _check_no_face(self, face_count, frame, annotated, frame_index):
        now = time.time()
        if face_count == 0:
            if self._no_face_start is None:
                self._no_face_start = now
            elapsed   = now - self._no_face_start
            remaining = max(0.0, NO_FACE_TIMEOUT_SEC - elapsed)
            cv2.putText(
                annotated,
                f"NO FACE DETECTED - violation in {remaining:.1f}s",
                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                BBOX_COLOR_VIOL, 2, cv2.LINE_AA,
            )
            if elapsed >= NO_FACE_TIMEOUT_SEC and not self._no_face_violated:
                path = capture_screenshot(frame, frame_index, "no_face")
                self.logger.log(
                    violation_type  = ViolationType.NO_FACE,
                    confidence      = 1.0,
                    screenshot_path = path,
                    extra           = {"elapsed_seconds": round(elapsed, 2)},
                )
                self._no_face_violated = True
        else:
            self._no_face_start    = None
            self._no_face_violated = False

    def _check_multiple_faces(self, face_count, frame, annotated, frame_index, detections):
        if face_count <= 1:
            return
        now = time.time()
        if now - self._multi_face_last_event < 3.0:
            return
        scores   = [d.categories[0].score for d in detections if d.categories]
        avg_conf = sum(scores) / len(scores) if scores else 0.9
        path     = capture_screenshot(frame, frame_index, "multi_face")
        self.logger.log(
            violation_type  = ViolationType.MULTI_FACE,
            confidence      = round(avg_conf, 4),
            screenshot_path = path,
            extra           = {"face_count": face_count},
        )
        self._multi_face_last_event = now
        cv2.putText(
            annotated,
            f"MULTIPLE FACES DETECTED ({face_count})",
            (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
            BBOX_COLOR_VIOL, 2, cv2.LINE_AA,
        )

    def _draw_detections(self, annotated, detections, shape):
        h, w = shape[:2]
        for i, det in enumerate(detections):
            bb   = det.bounding_box
            x1, y1 = bb.origin_x, bb.origin_y
            x2, y2 = x1 + bb.width, y1 + bb.height
            color  = BBOX_COLOR_OK if i == 0 else BBOX_COLOR_VIOL
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            score = det.categories[0].score if det.categories else 0.0
            cv2.putText(
                annotated, f"Face {i+1}  {score:.2f}", (x1, max(y1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA,
            )
