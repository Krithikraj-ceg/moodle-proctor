# AI Exam Proctoring System v3 — Complete Edition

Real-time AI-powered exam proctoring with **11 detection modules**.

---

## Full Module List

| # | Module | File | What it detects |
|---|--------|------|-----------------|
| 1 | Face Monitor | `face_monitor.py` | No face (5s) / multiple faces |
| 2 | Gaze Tracker | `gaze_tracking.py` | Head pose / looking away (3s) |
| 3 | Phone Detector | `phone_detection.py` | Mobile phones (YOLO) |
| 4 | Object Detector | `object_detection.py` | Books, laptops, remotes (YOLO) |
| 5 | Audio Monitor | `audio_monitor.py` | Suspicious sounds / talking |
| 6 | Blink Monitor | `blink_monitor.py` | Abnormal blink rate (reading notes) |
| 7 | Lip Monitor | `lip_movement.py` | Talking / whispering (3s) |
| 8 | Tab Monitor | `tab_monitor.py` | App / window switching |
| 9 | Lighting Monitor | `lighting_monitor.py` | Camera blocked / torch flash |
| 10 | Motion Detector | `motion_detector.py` | Background movement / person entering |
| 11 | Identity Verifier | `identity_verifier.py` | Candidate swap detection |

---

## Installation

```powershell
python -m venv .venv --without-pip
.venv\Scripts\python.exe -m ensurepip --upgrade
.venv\Scripts\pip3.exe install -r requirements.txt
```

---

## Configuration

Edit **`config.py`** to control everything:

```python
CANDIDATE_ID = "CANDIDATE-001"
EXAM_NAME    = "Mathematics Paper 1"
STRICTNESS   = "moderate"   # "strict" | "moderate" | "lenient"

# Toggle individual modules
ENABLE_AUDIO_MONITOR   = True
ENABLE_IDENTITY_VERIFY = True
# etc.
```

---

## Running

```powershell
cd ai_proctoring
C:\..\.venv\Scripts\python.exe main.py
```

Press **`q`** to quit. A **PDF report** is auto-generated on exit.

---

## Violation Types (13 total)

| Type | Trigger |
|------|---------|
| `NO_FACE` | No face for > 5s |
| `MULTIPLE_FACES` | 2+ faces visible |
| `LOOKING_AWAY` | Head turned away > 3s |
| `PHONE_DETECTED` | Mobile phone visible |
| `FORBIDDEN_OBJECT` | Book, laptop, remote detected |
| `AUDIO_VIOLATION` | Sustained sound / repeated spikes |
| `BLINK_ANOMALY` | < 3 or > 40 blinks/min |
| `TALKING_DETECTED` | Lips moving > 3s |
| `TAB_SWITCH` | 2+ app switches in 60s |
| `CAMERA_BLOCKED` | Darkness for > 3s |
| `LIGHTING_CHANGE` | Sudden brightness spike |
| `BACKGROUND_MOTION` | Movement behind candidate |
| `IDENTITY_MISMATCH` | Different person detected |

---

## Output

- `violations.jsonl` — structured JSON event log
- `screenshots/` — PNG snapshots of each violation
- `report_CANDIDATE_TIMESTAMP.pdf` — auto-generated on session end
