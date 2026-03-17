"""
Lighting Monitor
Detects sudden changes in ambient brightness — a common cheating signal:
  - Sudden BRIGHT spike → torch/flashlight pointed at notes/phone
  - Sudden DARK drop → candidate covering camera
  - Persistent darkness → camera blocked

Moderate mode: allows gradual changes (natural lighting), flags sudden spikes.
"""

import cv2
import time
import numpy as np
from collections import deque

from violation_logger import ViolationLogger, ViolationType
from utils import capture_screenshot


# ─── Tuning ───────────────────────────────────────────────────────────────────
HISTORY_LEN        = 60     # Rolling frame window for baseline
SPIKE_DELTA        = 40     # Brightness change in one step = spike
DARK_THRESHOLD     = 25     # Mean pixel value below this = too dark
DARK_DURATION_SEC  = 3.0    # Seconds of darkness before violation
EVENT_COOLDOWN_SEC = 8.0


class LightingMonitor:
    """Monitor ambient brightness for suspicious changes."""

    def __init__(self, logger: ViolationLogger):
        self.logger           = logger
        self._history         = deque(maxlen=HISTORY_LEN)
        self._dark_start      = None
        self._last_event_time = 0.0
        self.current_brightness = 0
        self.status           = "NORMAL"
        print("[LightingMonitor] Initialised.")

    def process(self, frame_bgr, annotated_bgr, frame_index: int) -> dict:
        gray       = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        self.current_brightness = int(brightness)

        self._check_camera_blocked(brightness, frame_bgr, annotated_bgr, frame_index)
        self._check_sudden_spike(brightness, frame_bgr, annotated_bgr, frame_index)

        self._history.append(brightness)
        self._draw_info(annotated_bgr, brightness)

        return {"brightness": self.current_brightness, "status": self.status}

    def _check_camera_blocked(self, brightness, frame, annotated, frame_index):
        now = time.time()
        if brightness < DARK_THRESHOLD:
            self.status = "DARK"
            if self._dark_start is None:
                self._dark_start = now
            elapsed = now - self._dark_start
            remaining = max(0.0, DARK_DURATION_SEC - elapsed)
            cv2.putText(annotated,
                        f"CAMERA BLOCKED? — violation in {remaining:.1f}s",
                        (20, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (0, 0, 180), 2, cv2.LINE_AA)
            if elapsed >= DARK_DURATION_SEC and now - self._last_event_time >= EVENT_COOLDOWN_SEC:
                path = capture_screenshot(frame, frame_index, "camera_blocked")
                self.logger.log(
                    violation_type  = ViolationType.CAMERA_BLOCKED,
                    confidence      = 0.95,
                    screenshot_path = path,
                    extra           = {"brightness": self.current_brightness,
                                       "elapsed_sec": round(elapsed, 2)},
                )
                self._last_event_time = now
        else:
            self._dark_start = None
            if self.status == "DARK":
                self.status = "NORMAL"

    def _check_sudden_spike(self, brightness, frame, annotated, frame_index):
        if len(self._history) < 10:
            return
        baseline = float(np.mean(list(self._history)[-10:]))
        delta    = abs(brightness - baseline)
        now      = time.time()

        if delta > SPIKE_DELTA and now - self._last_event_time >= EVENT_COOLDOWN_SEC:
            direction = "BRIGHT_SPIKE" if brightness > baseline else "DARK_SPIKE"
            self.status = direction
            path = capture_screenshot(frame, frame_index, "light_spike")
            self.logger.log(
                violation_type  = ViolationType.LIGHTING_CHANGE,
                confidence      = min(1.0, delta / (SPIKE_DELTA * 2)),
                screenshot_path = path,
                extra           = {"direction": direction,
                                   "delta": round(delta, 1),
                                   "brightness": self.current_brightness},
            )
            self._last_event_time = now

    def _draw_info(self, annotated, brightness):
        color = (0, 0, 200) if brightness < DARK_THRESHOLD else (0, 220, 80)
        cv2.putText(annotated,
                    f"Light: {int(brightness)}  [{self.status}]",
                    (annotated.shape[1]-260, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
