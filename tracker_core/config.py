import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple, List

import yaml


@dataclass
class AppConfig:
    camera_index: int
    frame_width: int
    frame_height: int

    enable_posture: bool
    enable_tts: bool
    enable_logging: bool

    distraction_yaw_deg: float
    distraction_min_seconds: float

    gaze_offset_threshold: float
    gaze_distraction_min_seconds: float

    drowsy_ear_threshold: float
    drowsy_min_seconds: float

    posture_slouch_angle_deg: float
    posture_min_seconds: float

    alert_cooldown_seconds: float
    tts_rate: int
    tts_volume: float

    log_dir: str
    log_frame_level: bool
    log_event_level: bool

    active_mode: str
    available_modes: List[str]


def _apply_overrides(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    out.update(overrides or {})
    return out


def load_config(path: str = "data/configs/default_config.yaml") -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    log_dir = raw.get("log_dir", "data/logs")
    os.makedirs(log_dir, exist_ok=True)

    active_mode = raw.get("active_mode", "Study")
    modes = raw.get("modes", {}) or {}
    available_modes = list(modes.keys()) if modes else ["Default"]

    effective = dict(raw)
    if active_mode in modes:
        effective = _apply_overrides(effective, modes[active_mode])

    return AppConfig(
        camera_index=effective.get("camera_index", 0),
        frame_width=effective.get("frame_width", 1280),
        frame_height=effective.get("frame_height", 720),

        enable_posture=effective.get("enable_posture", True),
        enable_tts=effective.get("enable_tts", True),
        enable_logging=effective.get("enable_logging", True),

        distraction_yaw_deg=effective.get("distraction_yaw_deg", 20.0),
        distraction_min_seconds=effective.get("distraction_min_seconds", 1.0),

        gaze_offset_threshold=effective.get("gaze_offset_threshold", 0.15),
        gaze_distraction_min_seconds=effective.get("gaze_distraction_min_seconds", 1.0),

        drowsy_ear_threshold=effective.get("drowsy_ear_threshold", 0.20),
        drowsy_min_seconds=effective.get("drowsy_min_seconds", 2.0),

        posture_slouch_angle_deg=effective.get("posture_slouch_angle_deg", 25.0),
        posture_min_seconds=effective.get("posture_min_seconds", 3.0),

        alert_cooldown_seconds=effective.get("alert_cooldown_seconds", 10.0),
        tts_rate=effective.get("tts_rate", 180),
        tts_volume=effective.get("tts_volume", 0.9),

        log_dir=log_dir,
        log_frame_level=effective.get("log_frame_level", True),
        log_event_level=effective.get("log_event_level", True),

        active_mode=active_mode,
        available_modes=available_modes,
    )