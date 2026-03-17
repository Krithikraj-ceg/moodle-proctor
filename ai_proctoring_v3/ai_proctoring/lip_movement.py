"""
Lip Movement Monitor
Analyses the mouth aspect ratio (MAR) from FaceMesh landmarks to detect
sustained talking or whispering — a strong indicator of cheating.

Moderate mode: flags only after lips move continuously for > 3 seconds.
"""

import cv2
import time
import numpy as np
from collections import deque

from violation_logger import ViolationLogger, ViolationType
from utils import capture_screenshot


# MediaPipe FaceLandmarker mouth landmark indices
# Upper lip: 13, Lower lip: 14, Left corner: 78, Right corner: 308
MOUTH_TOP    = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT   = 78
MOUTH_RIGHT  = 308

MAR_THRESHOLD      = 0.04   # Mouth Aspect Ratio — above this = mouth open
TALKING_SEC        = 3.0    # Must talk for this long before violation (moderate)
EVENT_COOLDOWN_SEC = 8.0
HISTORY_LEN        = 30     # Frames of MAR history for smoothing


class LipMovementMonitor:
    """Detect sustained lip/mouth movement (talking, whispering)."""

    def __init__(self, logger: ViolationLogger):
        self.logger          = logger
        self._talk_start     = None
        self._last_event_time = 0.0
        self._mar_history    = deque(maxlen=HISTORY_LEN)
        self.current_mar     = 0.0
        self.status          = "CLOSED"
        print("[LipMonitor] Initialised (MAR-based lip movement detection).")

    def process(self, landmarks, frame_bgr, annotated_bgr, frame_index, w, h) -> dict:
        if not landmarks:
            return {"lip_status": "NO_FACE", "mar": 0.0}

        mar = self._compute_mar(landmarks, w, h)
        self._mar_history.append(mar)
        smooth_mar       = float(np.mean(self._mar_history))
        self.current_mar = round(smooth_mar, 4)

        mouth_open = smooth_mar > MAR_THRESHOLD
        self.status = "TALKING" if mouth_open else "CLOSED"

        self._handle_talking(mouth_open, frame_bgr, annotated_bgr, frame_index, smooth_mar)
        self._draw_info(annotated_bgr, smooth_mar, mouth_open)

        return {"lip_status": self.status, "mar": self.current_mar}

    def _compute_mar(self, landmarks, w, h) -> float:
        def pt(idx):
            return np.array([landmarks[idx].x * w, landmarks[idx].y * h])

        top    = pt(MOUTH_TOP)
        bottom = pt(MOUTH_BOTTOM)
        left   = pt(MOUTH_LEFT)
        right  = pt(MOUTH_RIGHT)

        vertical   = np.linalg.norm(top - bottom)
        horizontal = np.linalg.norm(left - right) + 1e-6
        return vertical / horizontal

    def _handle_talking(self, mouth_open, frame, annotated, frame_index, mar):
        now = time.time()
        if mouth_open:
            if self._talk_start is None:
                self._talk_start = now
            elapsed   = now - self._talk_start
            remaining = max(0.0, TALKING_SEC - elapsed)
            cv2.putText(annotated,
                        f"LIP MOVEMENT — violation in {remaining:.1f}s",
                        (20, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (0, 80, 255), 2, cv2.LINE_AA)

            if elapsed >= TALKING_SEC and now - self._last_event_time >= EVENT_COOLDOWN_SEC:
                path = capture_screenshot(frame, frame_index, "talking")
                self.logger.log(
                    violation_type  = ViolationType.TALKING_DETECTED,
                    confidence      = min(1.0, mar / (MAR_THRESHOLD * 3)),
                    screenshot_path = path,
                    extra           = {"mar": round(mar, 4),
                                       "elapsed_sec": round(elapsed, 2)},
                )
                self._last_event_time = now
                self._talk_start      = None
        else:
            self._talk_start = None

    def _draw_info(self, annotated, mar, mouth_open):
        color = (0, 80, 255) if mouth_open else (0, 220, 80)
        cv2.putText(annotated,
                    f"Mouth: {'OPEN' if mouth_open else 'CLOSED'}  MAR:{mar:.3f}",
                    (annotated.shape[1]-300, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
