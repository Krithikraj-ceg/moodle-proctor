"""
Motion / Background Activity Detector
Detects movement in the background behind the candidate:
  - Another person entering the room
  - Objects being passed
  - Significant environmental movement

Uses frame differencing on the background region (outside face bounding box).
Moderate mode: flags only sustained or large-area background motion.
"""

import cv2
import time
import numpy as np
from collections import deque

from violation_logger import ViolationLogger, ViolationType
from utils import capture_screenshot


# ─── Tuning ───────────────────────────────────────────────────────────────────
MOTION_THRESH      = 25      # Pixel diff threshold
MOTION_AREA_PCT    = 0.08    # % of frame with motion to trigger flag
SUSTAINED_SEC      = 2.0     # Sustained background motion before violation
EVENT_COOLDOWN_SEC = 8.0
BLUR_KERNEL        = (21, 21)


class MotionDetector:
    """Detect background motion behind the candidate."""

    def __init__(self, logger: ViolationLogger):
        self.logger           = logger
        self._prev_gray       = None
        self._motion_start    = None
        self._last_event_time = 0.0
        self.motion_pct       = 0.0
        self.status           = "STILL"
        print("[MotionDetector] Initialised (frame-differencing background motion).")

    def process(self, frame_bgr, annotated_bgr, frame_index: int,
                face_bbox=None) -> dict:
        gray    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray    = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)
        h, w    = gray.shape

        if self._prev_gray is None:
            self._prev_gray = gray
            return {"motion_pct": 0.0, "status": "STILL"}

        # Mask out face region so candidate's own movement doesn't trigger
        mask = np.ones((h, w), dtype=np.uint8) * 255
        if face_bbox:
            x1, y1, x2, y2 = face_bbox
            # Expand face region by 50% to cover head movement
            pad_x = int((x2 - x1) * 0.5)
            pad_y = int((y2 - y1) * 0.5)
            x1m = max(0, x1 - pad_x)
            y1m = max(0, y1 - pad_y)
            x2m = min(w, x2 + pad_x)
            y2m = min(h, y2 + pad_y)
            mask[y1m:y2m, x1m:x2m] = 0

        diff    = cv2.absdiff(self._prev_gray, gray)
        diff    = cv2.bitwise_and(diff, diff, mask=mask)
        _, thresh = cv2.threshold(diff, MOTION_THRESH, 255, cv2.THRESH_BINARY)

        motion_pixels  = np.count_nonzero(thresh)
        total_pixels   = np.count_nonzero(mask)
        motion_pct     = motion_pixels / (total_pixels + 1e-6)
        self.motion_pct = round(float(motion_pct), 4)

        self._prev_gray = gray

        now = time.time()
        if motion_pct > MOTION_AREA_PCT:
            self.status = "MOTION"
            if self._motion_start is None:
                self._motion_start = now
            elapsed = now - self._motion_start

            # Overlay motion mask
            motion_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            motion_color[:, :] = [0, 0, 60]
            motion_color[thresh > 0] = [0, 60, 180]
            cv2.addWeighted(annotated_bgr, 0.85, motion_color, 0.15, 0, annotated_bgr)

            remaining = max(0.0, SUSTAINED_SEC - elapsed)
            cv2.putText(annotated_bgr,
                        f"BACKGROUND MOTION — {motion_pct*100:.1f}% — viol in {remaining:.1f}s",
                        (20, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 60, 255), 2, cv2.LINE_AA)

            if elapsed >= SUSTAINED_SEC and now - self._last_event_time >= EVENT_COOLDOWN_SEC:
                path = capture_screenshot(frame_bgr, frame_index, "bg_motion")
                self.logger.log(
                    violation_type  = ViolationType.BACKGROUND_MOTION,
                    confidence      = min(1.0, motion_pct / MOTION_AREA_PCT),
                    screenshot_path = path,
                    extra           = {"motion_pct": round(motion_pct * 100, 2),
                                       "elapsed_sec": round(elapsed, 2)},
                )
                self._last_event_time = now
                self._motion_start    = None
        else:
            self.status        = "STILL"
            self._motion_start = None

        return {"motion_pct": self.motion_pct, "status": self.status}
