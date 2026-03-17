"""
AI Proctoring System v3 — Complete Edition
Orchestrates ALL detection modules in a real-time webcam loop.

Detectors active:
  1.  Face presence & multi-face     (face_monitor)
  2.  Head pose / gaze               (gaze_tracking)
  3.  Phone detection                (phone_detection)
  4.  Forbidden objects              (object_detection)
  5.  Audio / sound                  (audio_monitor)
  6.  Blink rate anomaly             (blink_monitor)
  7.  Lip movement / talking         (lip_movement)
  8.  Tab / window switch            (tab_monitor)
  9.  Lighting change / camera block (lighting_monitor)
  10. Background motion              (motion_detector)
  11. Identity verification          (identity_verifier)
"""

import cv2
import time
import signal
import sys
from datetime import datetime

import config as C
from violation_logger  import ViolationLogger
from utils             import draw_status_overlay, ensure_directories

# ── Conditional imports based on config toggles ────────────────────────────────
if C.ENABLE_FACE_MONITOR:
    from face_monitor      import FaceMonitor
if C.ENABLE_GAZE_TRACKING:
    from gaze_tracking     import GazeTracker
if C.ENABLE_PHONE_DETECTION:
    from phone_detection   import PhoneDetector
if C.ENABLE_OBJECT_DETECT:
    from object_detection  import ObjectDetector
if C.ENABLE_AUDIO_MONITOR:
    from audio_monitor     import AudioMonitor
if C.ENABLE_BLINK_MONITOR:
    from blink_monitor     import BlinkMonitor
if C.ENABLE_LIP_MONITOR:
    from lip_movement      import LipMovementMonitor
if C.ENABLE_TAB_MONITOR:
    from tab_monitor       import TabMonitor
if C.ENABLE_LIGHTING_MONITOR:
    from lighting_monitor  import LightingMonitor
if C.ENABLE_MOTION_DETECT:
    from motion_detector   import MotionDetector
if C.ENABLE_IDENTITY_VERIFY:
    from identity_verifier import IdentityVerifier


def main():
    ensure_directories()
    session_start = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # ── Webcam setup ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(C.WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  C.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          C.TARGET_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    if not cap.isOpened():
        print("[FATAL] Cannot open webcam. Check device index.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  AI PROCTORING SYSTEM v3 — Complete Edition")
    print(f"  Candidate : {C.CANDIDATE_ID}")
    print(f"  Exam      : {C.EXAM_NAME}")
    print(f"  Strictness: {C.STRICTNESS.upper()}")
    print(f"{'='*60}\n")
    print("[INFO] Initialising AI modules ...")

    # ── Initialise all modules ────────────────────────────────────────────────
    logger           = ViolationLogger(C.LOG_FILE)
    face_monitor     = FaceMonitor(logger)     if C.ENABLE_FACE_MONITOR    else None
    gaze_tracker     = GazeTracker(logger)     if C.ENABLE_GAZE_TRACKING   else None
    phone_detector   = PhoneDetector(logger)   if C.ENABLE_PHONE_DETECTION else None
    object_detector  = ObjectDetector(logger)  if C.ENABLE_OBJECT_DETECT   else None
    audio_monitor    = AudioMonitor(logger)    if C.ENABLE_AUDIO_MONITOR   else None
    blink_monitor    = BlinkMonitor(logger)    if C.ENABLE_BLINK_MONITOR   else None
    lip_monitor      = LipMovementMonitor(logger) if C.ENABLE_LIP_MONITOR  else None
    tab_monitor      = TabMonitor(logger)      if C.ENABLE_TAB_MONITOR     else None
    lighting_monitor = LightingMonitor(logger) if C.ENABLE_LIGHTING_MONITOR else None
    motion_detector  = MotionDetector(logger)  if C.ENABLE_MOTION_DETECT   else None
    identity_verifier= IdentityVerifier(logger)if C.ENABLE_IDENTITY_VERIFY else None

    frame_count = 0
    fps_timer   = time.time()
    display_fps = 0.0
    last_face_bbox = None   # Shared between modules

    # ── Graceful shutdown ──────────────────────────────────────────────────────
    def _shutdown(sig=None, frame_sig=None):
        print("\n[INFO] Shutting down — generating session report ...")
        cap.release()
        cv2.destroyAllWindows()
        if audio_monitor:  audio_monitor.stop()
        if tab_monitor:    tab_monitor.stop()
        logger.close()

        try:
            from report_generator import generate_report
            report_path = generate_report(
                log_path      = C.LOG_FILE,
                output_path   = f"report_{C.CANDIDATE_ID}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                session_start = session_start,
                candidate_id  = C.CANDIDATE_ID,
                exam_name     = C.EXAM_NAME,
                institution   = C.INSTITUTION,
                strictness    = C.STRICTNESS,
            )
            print(f"[INFO] PDF report saved: {report_path}")
        except Exception as exc:
            print(f"[WARN] PDF generation failed: {exc}")
            print("       Run: pip install reportlab")
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print("[INFO] All modules ready. Press 'q' to quit and generate report.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame_count += 1
        annotated    = frame.copy()
        h, w         = frame.shape[:2]

        # FPS counter
        if frame_count % 30 == 0:
            elapsed     = time.time() - fps_timer
            display_fps = 30.0 / elapsed if elapsed > 0 else 0.0
            fps_timer   = time.time()

        # ── 1. Face detection ──────────────────────────────────────────────
        face_result = {}
        if face_monitor:
            face_result = face_monitor.process(frame, annotated, frame_count)
            # Extract face bbox for motion detector masking
            # (FaceMonitor exposes last bbox via attribute if set)
            last_face_bbox = getattr(face_monitor, '_last_bbox', None)

        # ── 2. Gaze + landmarks (shared with blink/lip/identity) ──────────
        gaze_result = {}
        landmarks   = None
        if gaze_tracker:
            gaze_result = gaze_tracker.process(frame, annotated, frame_count)
            landmarks   = gaze_result.get("landmarks")

        # ── 3. Phone detection ────────────────────────────────────────────
        phone_result = {}
        if phone_detector:
            phone_result = phone_detector.process(frame, annotated, frame_count)

        # ── 4. Forbidden objects ──────────────────────────────────────────
        object_result = {}
        if object_detector:
            object_result = object_detector.process(frame, annotated, frame_count)

        # ── 5. Blink rate ─────────────────────────────────────────────────
        blink_result = {}
        if blink_monitor and landmarks:
            blink_result = blink_monitor.process(landmarks, frame, annotated,
                                                  frame_count, w, h)

        # ── 6. Lip movement / talking ─────────────────────────────────────
        lip_result = {}
        if lip_monitor and landmarks:
            lip_result = lip_monitor.process(landmarks, frame, annotated,
                                              frame_count, w, h)

        # ── 7. Lighting ───────────────────────────────────────────────────
        light_result = {}
        if lighting_monitor:
            light_result = lighting_monitor.process(frame, annotated, frame_count)

        # ── 8. Background motion ──────────────────────────────────────────
        motion_result = {}
        if motion_detector:
            motion_result = motion_detector.process(frame, annotated,
                                                     frame_count, last_face_bbox)

        # ── 9. Identity verification ──────────────────────────────────────
        identity_result = {}
        if identity_verifier and landmarks:
            identity_result = identity_verifier.process(landmarks, frame, annotated,
                                                         frame_count, w, h)

        # ── HUD overlay ───────────────────────────────────────────────────
        if C.SHOW_DEBUG_INFO:
            status = {
                "fps":        display_fps,
                "frame":      frame_count,
                "faces":      face_result.get("face_count", 0),
                "gaze":       gaze_result.get("gaze_status", "N/A"),
                "phone":      phone_result.get("phone_detected", False),
                "objects":    object_result.get("count", 0),
                "audio":      getattr(audio_monitor,  "status",         "OFF"),
                "blink_bpm":  blink_result.get("blink_rate", 0.0),
                "lip":        lip_result.get("lip_status", "N/A"),
                "tab":        getattr(tab_monitor,     "status",         "N/A"),
                "light":      light_result.get("status", "N/A"),
                "motion":     motion_result.get("status", "N/A"),
                "identity":   identity_result.get("identity_status", "N/A"),
                "violations": logger.total_violations,
                "timestamp":  datetime.now().strftime("%H:%M:%S"),
            }
            draw_status_overlay(annotated, status)

        if C.SHOW_PREVIEW:
            cv2.imshow("AI Proctoring System v3", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                _shutdown()

    cap.release()
    cv2.destroyAllWindows()
    logger.close()


if __name__ == "__main__":
    main()
