"""Utility helpers — screenshot capture and full HUD overlay."""

import cv2, os
from datetime import datetime

SCREENSHOTS_DIR = "screenshots"

def ensure_directories():
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
    print(f"[Utils] Screenshot directory: '{SCREENSHOTS_DIR}/'")

def capture_screenshot(frame, frame_index: int, tag: str = "") -> str:
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix   = f"_{tag}" if tag else ""
    filename = f"frame_{frame_index:06d}{suffix}_{ts}.png"
    path     = os.path.join(SCREENSHOTS_DIR, filename)
    cv2.imwrite(path, frame)
    return path

def draw_status_overlay(frame, status: dict):
    lines = [
        f"  FPS        : {status.get('fps', 0):.1f}",
        f"  Frame      : {status.get('frame', 0)}",
        f"  Faces      : {status.get('faces', 0)}",
        f"  Gaze       : {status.get('gaze', 'N/A')}",
        f"  Phone      : {'YES !' if status.get('phone') else 'No'}",
        f"  Objects    : {status.get('objects', 0)} found",
        f"  Audio      : {status.get('audio', 'N/A')}",
        f"  Blink BPM  : {status.get('blink_bpm', 0.0):.1f}",
        f"  Lips       : {status.get('lip', 'N/A')}",
        f"  Tab        : {status.get('tab', 'N/A')}",
        f"  Light      : {status.get('light', 'N/A')}",
        f"  Motion     : {status.get('motion', 'N/A')}",
        f"  Identity   : {status.get('identity', 'N/A')}",
        f"  Violations : {status.get('violations', 0)}",
        f"  Time       : {status.get('timestamp', '')}",
    ]

    panel_w, line_h, pad = 290, 21, 10
    panel_h = line_h * len(lines) + pad * 2 + 26

    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    cv2.putText(frame, "AI PROCTORING SYSTEM v3",
                (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (100, 200, 255), 1, cv2.LINE_AA)
    cv2.line(frame, (10, 32), (10 + panel_w - 4, 32), (60, 60, 60), 1)

    ALERT_KEYWORDS = ["YES", "SWITCHED", "SOUND", "TALKING", "MISMATCH",
                      "MOTION", "DARK", "SPIKE", "LOW_BLINK", "HIGH_BLINK"]

    for i, line in enumerate(lines):
        y     = 32 + pad + (i + 1) * line_h
        color = (0, 220, 80)
        if any(k in line for k in ALERT_KEYWORDS):
            color = (50, 50, 255)
        elif "Violations" in line and status.get("violations", 0) > 0:
            color = (0, 140, 255)
        elif "ENROLLING" in line:
            color = (255, 200, 0)
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
