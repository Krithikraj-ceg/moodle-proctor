"""
Identity Verifier
Captures a reference face embedding at session start.
Periodically compares the live face to the reference to detect
candidate swapping — someone else sitting the exam.

Uses face landmark geometry as a lightweight embedding (no GPU needed).
For production, swap with a proper face recognition model (DeepFace/InsightFace).
"""

import cv2
import time
import numpy as np

from violation_logger import ViolationLogger, ViolationType
from utils import capture_screenshot


# ─── Tuning ───────────────────────────────────────────────────────────────────
ENROLL_FRAMES      = 30     # Frames averaged for stable reference embedding
CHECK_INTERVAL_SEC = 10.0   # How often to verify identity
SIMILARITY_THRESH  = 0.75   # Below this cosine similarity = different person
EVENT_COOLDOWN_SEC = 20.0


# Key landmark indices for identity embedding (eyes, nose, mouth corners)
IDENTITY_LANDMARKS = [
    33, 263,    # Eye centres (left, right)
    1,          # Nose tip
    61, 291,    # Mouth corners
    199,        # Chin
    10,         # Forehead centre
]


class IdentityVerifier:
    """Enroll a reference face at session start and verify periodically."""

    def __init__(self, logger: ViolationLogger):
        self.logger           = logger
        self._reference       = None   # Normalised landmark embedding
        self._enroll_buffer   = []
        self._enrolled        = False
        self._last_check_time = 0.0
        self._last_event_time = 0.0
        self.status           = "ENROLLING"
        self.similarity       = 1.0
        print("[IdentityVerifier] Enrolling reference face — please look at camera ...")

    def process(self, landmarks, frame_bgr, annotated_bgr, frame_index, w, h) -> dict:
        if not landmarks:
            return {"identity_status": self.status, "similarity": self.similarity}

        embedding = self._extract_embedding(landmarks, w, h)

        if not self._enrolled:
            self._enroll(embedding, annotated_bgr, frame_index)
        else:
            self._verify(embedding, frame_bgr, annotated_bgr, frame_index)

        self._draw_info(annotated_bgr)
        return {"identity_status": self.status, "similarity": self.similarity}

    def _extract_embedding(self, landmarks, w, h) -> np.ndarray:
        pts = np.array([
            [landmarks[i].x * w, landmarks[i].y * h]
            for i in IDENTITY_LANDMARKS
        ], dtype=np.float64)

        # Normalise: translate to centroid, scale by face width
        centroid = pts.mean(axis=0)
        pts      = pts - centroid
        scale    = np.linalg.norm(pts[0] - pts[1]) + 1e-6   # Eye distance
        pts      = pts / scale
        return pts.flatten()

    def _enroll(self, embedding, annotated, frame_index):
        self._enroll_buffer.append(embedding)
        progress = len(self._enroll_buffer) / ENROLL_FRAMES * 100
        cv2.putText(annotated,
                    f"ENROLLING FACE: {int(progress)}%",
                    (20, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 200, 0), 2, cv2.LINE_AA)

        if len(self._enroll_buffer) >= ENROLL_FRAMES:
            self._reference = np.mean(self._enroll_buffer, axis=0)
            self._enrolled  = True
            self.status     = "VERIFIED"
            print("[IdentityVerifier] Reference face enrolled successfully.")

    def _verify(self, embedding, frame, annotated, frame_index):
        now = time.time()
        if now - self._last_check_time < CHECK_INTERVAL_SEC:
            return
        self._last_check_time = now

        sim = self._cosine_similarity(embedding, self._reference)
        self.similarity = round(float(sim), 3)

        if sim < SIMILARITY_THRESH:
            self.status = "MISMATCH"
            if now - self._last_event_time >= EVENT_COOLDOWN_SEC:
                path = capture_screenshot(frame, frame_index, "identity_mismatch")
                self.logger.log(
                    violation_type  = ViolationType.IDENTITY_MISMATCH,
                    confidence      = round(1.0 - sim, 4),
                    screenshot_path = path,
                    extra           = {"similarity": self.similarity,
                                       "threshold":  SIMILARITY_THRESH},
                )
                self._last_event_time = now
        else:
            self.status = "VERIFIED"

    def _cosine_similarity(self, a, b) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-6
        return float(np.dot(a, b) / denom)

    def _draw_info(self, annotated):
        color = (0, 220, 80) if self.status == "VERIFIED" else \
                (255, 200, 0) if self.status == "ENROLLING" else (0, 0, 220)
        cv2.putText(annotated,
                    f"ID: {self.status}  sim={self.similarity:.2f}",
                    (annotated.shape[1]-280, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
