"""
Violation Logger — all violation types including new ones.
"""

import json, os, threading
from datetime import datetime
from enum import Enum

LOG_FILE = "violations.jsonl"

class ViolationType(str, Enum):
    # Original
    NO_FACE          = "NO_FACE"
    MULTI_FACE       = "MULTIPLE_FACES"
    LOOKING_AWAY     = "LOOKING_AWAY"
    PHONE_DETECTED   = "PHONE_DETECTED"
    # v2
    AUDIO_VIOLATION  = "AUDIO_VIOLATION"
    BLINK_ANOMALY    = "BLINK_ANOMALY"
    TAB_SWITCH       = "TAB_SWITCH"
    # v3 (new)
    FORBIDDEN_OBJECT = "FORBIDDEN_OBJECT"
    TALKING_DETECTED = "TALKING_DETECTED"
    CAMERA_BLOCKED   = "CAMERA_BLOCKED"
    LIGHTING_CHANGE  = "LIGHTING_CHANGE"
    BACKGROUND_MOTION= "BACKGROUND_MOTION"
    IDENTITY_MISMATCH= "IDENTITY_MISMATCH"

_CLR = {
    ViolationType.NO_FACE:           "\033[91m",
    ViolationType.MULTI_FACE:        "\033[93m",
    ViolationType.LOOKING_AWAY:      "\033[94m",
    ViolationType.PHONE_DETECTED:    "\033[95m",
    ViolationType.AUDIO_VIOLATION:   "\033[96m",
    ViolationType.BLINK_ANOMALY:     "\033[33m",
    ViolationType.TAB_SWITCH:        "\033[35m",
    ViolationType.FORBIDDEN_OBJECT:  "\033[91m",
    ViolationType.TALKING_DETECTED:  "\033[93m",
    ViolationType.CAMERA_BLOCKED:    "\033[91m",
    ViolationType.LIGHTING_CHANGE:   "\033[33m",
    ViolationType.BACKGROUND_MOTION: "\033[36m",
    ViolationType.IDENTITY_MISMATCH: "\033[91m",
}
_RESET = "\033[0m"

class ViolationLogger:
    def __init__(self, log_path=LOG_FILE):
        self._log_path        = log_path
        self._lock            = threading.Lock()
        self._file            = open(log_path, "a", encoding="utf-8")
        self.total_violations = 0
        print(f"[ViolationLogger] Logging to '{log_path}'.")

    def log(self, violation_type, confidence, screenshot_path, extra=None) -> dict:
        event = {
            "timestamp":       datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "violation_type":  violation_type.value,
            "confidence":      round(confidence, 4),
            "screenshot_path": screenshot_path,
        }
        if extra:
            event["extra"] = extra
        with self._lock:
            self._file.write(json.dumps(event) + "\n")
            self._file.flush()
            self.total_violations += 1
        self._print_event(event, violation_type)
        return event

    def close(self):
        with self._lock:
            if not self._file.closed:
                self._file.close()
        print(f"[ViolationLogger] Closed. Total violations: {self.total_violations}")

    def read_all(self) -> list:
        events = []
        if not os.path.exists(self._log_path):
            return events
        with open(self._log_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try: events.append(json.loads(line))
                    except: pass
        return events

    def _print_event(self, event, vtype):
        colour = _CLR.get(vtype, "")
        print(f"{colour}[VIOLATION #{self.total_violations}] "
              f"{event['timestamp']}  |  {event['violation_type']}  |  "
              f"conf={event['confidence']:.2f}  |  {event['screenshot_path']}{_RESET}")
