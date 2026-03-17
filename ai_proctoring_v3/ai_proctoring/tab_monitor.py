"""
Tab / Window Switch Monitor
Detects when the candidate switches away from the exam window.

Uses Windows API (pywin32) on Windows, and xdotool / subprocess on Linux.
Moderate mode: allows 1 accidental switch, flags on 2nd+ within 60 s.
"""

import time
import threading
import platform
from collections import deque

from violation_logger import ViolationLogger, ViolationType
from utils import capture_screenshot


# ─── Tuning ───────────────────────────────────────────────────────────────────
POLL_INTERVAL_SEC  = 0.5    # How often to check active window
SWITCH_WINDOW_SEC  = 60.0   # Rolling window to count switches
SWITCH_LIMIT       = 2      # Moderate: allow 1 slip, flag on 2nd
EVENT_COOLDOWN_SEC = 10.0
EXAM_KEYWORDS      = []     # Optional: list of strings that must appear in
                            # the exam window title e.g. ["Exam", "Quiz"]
                            # Leave empty to monitor ANY focus loss


class TabMonitor:
    """Poll the active window and flag tab/app switches."""

    def __init__(self, logger: ViolationLogger):
        self.logger         = logger
        self._running       = False
        self._thread        = None
        self._switch_times  = deque()
        self._last_event_time = 0.0
        self._last_window   = ""
        self.switch_count   = 0
        self.status         = "FOCUSED"
        self._os            = platform.system()

        self._start()

    # ── Public ────────────────────────────────────────────────────────────────

    def stop(self):
        self._running = False

    # ── Private ───────────────────────────────────────────────────────────────

    def _start(self):
        if self._os not in ("Windows", "Linux"):
            print(f"[TabMonitor] Unsupported OS '{self._os}'. Tab monitoring disabled.")
            return
        try:
            self._get_active_window()   # Test the call
            self._running = True
            self._thread  = threading.Thread(target=self._poll_loop, daemon=True)
            self._thread.start()
            print(f"[TabMonitor] Window focus monitoring started ({self._os}).")
        except Exception as exc:
            print(f"[TabMonitor] WARNING: Cannot monitor windows — {exc}\n"
                  "  Windows: pip install pywin32\n"
                  "  Linux:   sudo apt install xdotool")

    def _get_active_window(self) -> str:
        if self._os == "Windows":
            import win32gui
            hwnd  = win32gui.GetForegroundWindow()
            return win32gui.GetWindowText(hwnd)
        elif self._os == "Linux":
            import subprocess
            result = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowname"],
                capture_output=True, text=True, timeout=1,
            )
            return result.stdout.strip()
        return ""

    def _poll_loop(self):
        # Capture the initial window as the "exam window"
        time.sleep(1.0)
        self._last_window = self._get_active_window()

        while self._running:
            time.sleep(POLL_INTERVAL_SEC)
            try:
                current = self._get_active_window()
                if current and current != self._last_window:
                    self._on_switch(self._last_window, current)
                    self._last_window = current
            except Exception:
                pass

    def _on_switch(self, from_win: str, to_win: str):
        now = time.time()
        self.switch_count += 1
        self.status = "SWITCHED"

        self._switch_times.append(now)
        while self._switch_times and now - self._switch_times[0] > SWITCH_WINDOW_SEC:
            self._switch_times.popleft()

        recent_switches = len(self._switch_times)

        print(f"[TabMonitor] Window switched: '{from_win}' → '{to_win}' "
              f"(recent: {recent_switches})")

        # Moderate: only fire violation after SWITCH_LIMIT switches in window
        if recent_switches >= SWITCH_LIMIT and now - self._last_event_time >= EVENT_COOLDOWN_SEC:
            self.logger.log(
                violation_type  = ViolationType.TAB_SWITCH,
                confidence      = min(1.0, recent_switches / SWITCH_LIMIT),
                screenshot_path = "N/A (window event)",
                extra           = {
                    "from_window":     from_win,
                    "to_window":       to_win,
                    "switches_in_window": recent_switches,
                    "total_switches":  self.switch_count,
                },
            )
            self._last_event_time = now
            self._switch_times.clear()
