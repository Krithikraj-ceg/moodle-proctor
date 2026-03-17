"""
Audio Monitor Module
Listens to the microphone in a background thread.
Flags suspicious sounds (talking, loud noise) that persist beyond a threshold.

Moderate mode:
  - Single short sound  → warning only
  - Sustained sound > 2 s OR 3+ spikes in 10 s → violation
"""

import time
import threading
import numpy as np
from collections import deque
from datetime import datetime

from violation_logger import ViolationLogger, ViolationType
from utils import capture_screenshot


# ─── Tuning ───────────────────────────────────────────────────────────────────
SAMPLE_RATE        = 16000
CHUNK_SIZE         = 1024
SOUND_THRESHOLD    = 1500    # RMS amplitude — raise if too sensitive
SUSTAINED_SEC      = 2.0     # Seconds of continuous sound before violation
SPIKE_WINDOW_SEC   = 10.0    # Window to count repeated spikes
SPIKE_COUNT_LIMIT  = 3       # N spikes in window → violation (moderate)
EVENT_COOLDOWN_SEC = 6.0


class AudioMonitor:
    """Background microphone monitor."""

    def __init__(self, logger: ViolationLogger):
        self.logger   = logger
        self._running = False
        self._thread  = None
        self._stream  = None
        self._pa      = None

        # State
        self._sound_start     = None
        self._spike_times     = deque()
        self._last_event_time = 0.0
        self.current_rms      = 0
        self.status           = "QUIET"

        self._try_start()

    # ── Public ────────────────────────────────────────────────────────────────

    def stop(self):
        self._running = False
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
        if self._pa:
            try:
                self._pa.terminate()
            except Exception:
                pass

    # ── Private ───────────────────────────────────────────────────────────────

    def _try_start(self):
        try:
            import pyaudio
            self._pa     = pyaudio.PyAudio()
            self._stream = self._pa.open(
                format            = pyaudio.paInt16,
                channels          = 1,
                rate              = SAMPLE_RATE,
                input             = True,
                frames_per_buffer = CHUNK_SIZE,
            )
            self._running = True
            self._thread  = threading.Thread(target=self._listen_loop, daemon=True)
            self._thread.start()
            print("[AudioMonitor] Microphone monitoring started.")
        except ImportError:
            print("[AudioMonitor] WARNING: pyaudio not installed. "
                  "Run: pip install pyaudio")
        except Exception as exc:
            print(f"[AudioMonitor] WARNING: Cannot open microphone — {exc}")

    def _listen_loop(self):
        while self._running:
            try:
                raw  = self._stream.read(CHUNK_SIZE, exception_on_overflow=False)
                data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                rms  = float(np.sqrt(np.mean(data ** 2)))
                self.current_rms = int(rms)
                self._evaluate(rms)
            except Exception:
                time.sleep(0.05)

    def _evaluate(self, rms: float):
        now   = time.time()
        loud  = rms > SOUND_THRESHOLD

        if loud:
            self.status = "SOUND_DETECTED"
            # Track spike times (moderate: count repeated bursts)
            self._spike_times.append(now)
            # Prune old spikes outside window
            while self._spike_times and now - self._spike_times[0] > SPIKE_WINDOW_SEC:
                self._spike_times.popleft()

            if self._sound_start is None:
                self._sound_start = now

            elapsed = now - self._sound_start

            # Violation condition (moderate): sustained OR repeated spikes
            sustained_viol = elapsed >= SUSTAINED_SEC
            spike_viol     = len(self._spike_times) >= SPIKE_COUNT_LIMIT

            if (sustained_viol or spike_viol) and (now - self._last_event_time >= EVENT_COOLDOWN_SEC):
                reason = "sustained_sound" if sustained_viol else "repeated_spikes"
                self.logger.log(
                    violation_type  = ViolationType.AUDIO_VIOLATION,
                    confidence      = min(1.0, rms / (SOUND_THRESHOLD * 2)),
                    screenshot_path = "N/A (audio event)",
                    extra           = {
                        "reason":       reason,
                        "rms":          self.current_rms,
                        "spike_count":  len(self._spike_times),
                        "elapsed_sec":  round(elapsed, 2),
                    },
                )
                self._last_event_time = now
                self._sound_start     = None
                self._spike_times.clear()
        else:
            self.status       = "QUIET"
            self._sound_start = None
