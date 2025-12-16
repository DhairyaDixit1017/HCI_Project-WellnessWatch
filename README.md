# HCI_Project-WellnessWatch
WellnessWatch is a multimodal HCI system that monitors **attention**, **drowsiness**, and **posture** from a webcam feed and provides real-time feedback via on-screen HUD and optional text-to-speech (TTS).

## Key features
- Real-time webcam monitoring (face + pose landmarks).
- System states: `OK`, `DISTRACTED`, `DROWSY`, `POSTURE_RISK`.
- Preset “modes” (e.g., Study/Office/HighFocus) that adjust thresholds/cooldowns without changing the UI.
- Two-level logging for HCI evaluation:
  - Frame-level logs (`frames_*.csv`)
  - Event-level logs (`events_*.csv`) for transitions, alerts, and user actions.

---

## Quick start

### 1) Install

pip install -r requirements.txt

### 2) Run

python main.py

### 3) Controls
- `m` = cycle mode (Study → Office → HighFocus)
- `c` = run calibration
- `q` = quit

---

## Repository structure

```
WellnessWatch/
├─ main.py
├─ tracker_core/
│ ├─ config.py
│ ├─ gaze_tracker.py
│ ├─ mode_manager.py
│ └─ angle_buffer.py
├─ wellness/
│ ├─ attention_inference.py
│ ├─ posture_module.py
│ ├─ feedback.py
│ ├─ calibration.py
│ └─ logging_utils.py
├─ data/
│ ├─ configs/
│ │ └─ default_config.yaml
│ └─ logs/
│ ├─ frames_YYYYMMDD_HHMMSS.csv
│ └─ events_YYYYMMDD_HHMMSS.csv
└─ requirements.txt
```