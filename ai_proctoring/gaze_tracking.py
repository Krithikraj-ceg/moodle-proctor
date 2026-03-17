"""
Gaze Tracking Module
Estimates head pose using MediaPipe FaceLandmarker (Tasks API, mediapipe>=0.10).
Generates a LOOKING_AWAY violation if the candidate looks away for > 3 seconds.
"""

import cv2
import time
import urllib.request
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from violation_logger import ViolationLogger, ViolationType
from utils import capture_screenshot


# ─── Tuning Constants ─────────────────────────────────────────────────────────
LOOK_AWAY_TIMEOUT_SEC  = 3.0
YAW_THRESHOLD_DEG      = 25.0
PITCH_UP_THRESHOLD_DEG = 20.0   # Looking UP is suspicious
PITCH_DOWN_THRESHOLD_DEG = 40.0 # Looking DOWN is allowed (writing) - only flag extreme down

PNP_LANDMARK_IDS = [1, 152, 263, 33, 287, 57]

MODEL_POINTS_3D = np.array([
    [0.0,    0.0,   0.0  ],
    [0.0,  -63.6, -12.5  ],
    [-43.3,  32.7, -26.0 ],
    [43.3,   32.7, -26.0 ],
    [-28.9, -28.9, -24.1 ],
    [28.9,  -28.9, -24.1 ],
], dtype=np.float64)

MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = "face_landmarker.task"


def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("[GazeTracker] Downloading face landmarker model (~30 MB) ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"[GazeTracker] Model saved to '{MODEL_PATH}'.")


class GazeTracker:
    """Estimate head pose and detect look-away behaviour."""

    def __init__(self, logger: ViolationLogger):
        self.logger = logger
        _ensure_model()

        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options      = mp_vision.FaceLandmarkerOptions(
            base_options                  = base_options,
            num_faces                     = 1,
            min_face_detection_confidence = 0.6,
            min_tracking_confidence       = 0.6,
        )
        self.face_mesh = mp_vision.FaceLandmarker.create_from_options(options)

        # State
        self._look_away_start   = None
        self._look_away_violated = False
        self._last_event_time   = 0.0

        print("[GazeTracker] Initialised (MediaPipe Tasks FaceLandmarker + PnP solver).")

    # ── Public ────────────────────────────────────────────────────────────────

    def process(self, frame_bgr, annotated_bgr, frame_index: int) -> dict:
        h, w   = frame_bgr.shape[:2]
        rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.face_mesh.detect(mp_img)

        if not result.face_landmarks:
            return {"gaze_status": "NO_FACE", "yaw": None, "pitch": None}

        landmarks = result.face_landmarks[0]
        yaw, pitch = self._estimate_head_pose(landmarks, w, h, annotated_bgr)

        if yaw is None:
            return {"gaze_status": "POSE_FAIL", "yaw": None, "pitch": None}

        looking_away = (
            abs(yaw) > YAW_THRESHOLD_DEG or      # looking left/right
            pitch < -PITCH_UP_THRESHOLD_DEG or   # looking up
            pitch > PITCH_DOWN_THRESHOLD_DEG      # extreme down (phone under desk etc)
        )

        direction = self._gaze_direction(yaw, pitch)
        self._handle_look_away(looking_away, frame_bgr, annotated_bgr,
                               frame_index, yaw, pitch, direction)
        self._draw_pose_info(annotated_bgr, yaw, pitch, direction, looking_away)

        return {
            "gaze_status":  direction,
            "yaw":          round(yaw,   2),
            "pitch":        round(pitch, 2),
            "looking_away": looking_away,
            "landmarks":    landmarks,    # passed to BlinkMonitor
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _estimate_head_pose(self, landmarks, w, h, annotated):
        image_pts = np.array([
            [landmarks[i].x * w, landmarks[i].y * h]
            for i in PNP_LANDMARK_IDS
        ], dtype=np.float64)

        focal   = w
        cam_mat = np.array([
            [focal, 0,     w / 2],
            [0,     focal, h / 2],
            [0,     0,     1    ],
        ], dtype=np.float64)
        dist_coeff = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            MODEL_POINTS_3D, image_pts, cam_mat, dist_coeff,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None, None

        rmat, _ = cv2.Rodrigues(rvec)
        proj_mat = np.hstack([rmat, tvec])
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj_mat)

        euler = euler.flatten()
        pitch = float(euler[0])
        yaw   = float(euler[1])

        nose_end_3d = np.array([[0.0, 0.0, 40.0]])
        nose_end_2d, _ = cv2.projectPoints(nose_end_3d, rvec, tvec, cam_mat, dist_coeff)
        nose_tip = (int(image_pts[0][0]), int(image_pts[0][1]))
        nose_end = (int(nose_end_2d[0][0][0]), int(nose_end_2d[0][0][1]))
        cv2.arrowedLine(annotated, nose_tip, nose_end, (255, 200, 0), 2, tipLength=0.3)

        return yaw, pitch

    def _gaze_direction(self, yaw, pitch) -> str:
        # Looking down is normal (writing on paper) — only flag left/right/up/extreme-down
        if (abs(yaw) <= YAW_THRESHOLD_DEG and
                pitch >= -PITCH_UP_THRESHOLD_DEG and
                pitch <= PITCH_DOWN_THRESHOLD_DEG):
            return "FORWARD"
        parts = []
        if pitch < -PITCH_UP_THRESHOLD_DEG:
            parts.append("UP")
        elif pitch > PITCH_DOWN_THRESHOLD_DEG:
            parts.append("EXTREME_DOWN")  # Suspicious (hiding phone under desk)
        if yaw < -YAW_THRESHOLD_DEG:
            parts.append("LEFT")
        elif yaw > YAW_THRESHOLD_DEG:
            parts.append("RIGHT")
        return "_".join(parts) if parts else "AWAY"

    def _handle_look_away(self, looking_away, frame, annotated, frame_index,
                          yaw, pitch, direction):
        now = time.time()
        if looking_away:
            if self._look_away_start is None:
                self._look_away_start = now
            elapsed   = now - self._look_away_start
            remaining = max(0.0, LOOK_AWAY_TIMEOUT_SEC - elapsed)
            cv2.putText(
                annotated,
                f"LOOKING {direction} - violation in {remaining:.1f}s",
                (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (0, 0, 220), 2, cv2.LINE_AA,
            )
            if elapsed >= LOOK_AWAY_TIMEOUT_SEC and not self._look_away_violated:
                if now - self._last_event_time >= 5.0:
                    path = capture_screenshot(frame, frame_index, "look_away")
                    self.logger.log(
                        violation_type  = ViolationType.LOOKING_AWAY,
                        confidence      = min(1.0, elapsed / LOOK_AWAY_TIMEOUT_SEC),
                        screenshot_path = path,
                        extra           = {
                            "direction":       direction,
                            "yaw_deg":         round(yaw,   2),
                            "pitch_deg":       round(pitch, 2),
                            "elapsed_seconds": round(elapsed, 2),
                        },
                    )
                    self._last_event_time    = now
                    self._look_away_violated = True
        else:
            self._look_away_start    = None
            self._look_away_violated = False

    def _draw_pose_info(self, annotated, yaw, pitch, direction, looking_away):
        color = (0, 0, 220) if looking_away else (0, 220, 80)
        for i, text in enumerate([
            f"Yaw:   {yaw:+.1f} deg",
            f"Pitch: {pitch:+.1f} deg",
            f"Gaze:  {direction}",
        ]):
            cv2.putText(
                annotated, text,
                (annotated.shape[1] - 260, 40 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA,
            )
