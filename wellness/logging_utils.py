import csv
import os
import time
import json

from tracker_core.config import AppConfig
from tracker_core.gaze_tracker import GazeFrameData
from wellness.posture_module import PostureData
from wellness.attention_inference import StateOutput


class EventLogger:
    def __init__(self, config: AppConfig):
        self.config = config
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.frame_log_path = os.path.join(config.log_dir, f"frames_{ts}.csv")
        self.event_log_path = os.path.join(config.log_dir, f"events_{ts}.csv")

        if config.log_frame_level:
            with open(self.frame_log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "mode",
                        "face_detected",
                        "ear_avg",
                        "head_yaw_deg",
                        "head_pitch_deg",
                        "gaze_offset_x",
                        "posture_state",
                        "torso_tilt_deg",
                        "lean_score",
                        "forward_head_score",
                        "system_state",
                        "reasons",
                    ]
                )

        if config.log_event_level:
            with open(self.event_log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "event_type", "details", "details_json"])

    def log_frame(
        self,
        config: AppConfig,
        gaze: GazeFrameData,
        posture: PostureData,
        state_output: StateOutput,
    ):
        if not self.config.log_frame_level:
            return

        t = time.time()
        with open(self.frame_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    t,
                    getattr(config, "active_mode", ""),
                    gaze.face_detected,
                    gaze.ear_avg,
                    gaze.head_yaw_deg,
                    gaze.head_pitch_deg,
                    gaze.gaze_offset_x,
                    posture.posture_state,
                    posture.torso_tilt_deg,
                    posture.lean_score,
                    posture.forward_head_score,
                    state_output.state.name,
                    "|".join(state_output.reasons),
                ]
            )

    def log_event(self, event_type: str, details: str = "", details_obj=None):
        if not self.config.log_event_level:
            return

        t = time.time()
        details_json = json.dumps(details_obj) if details_obj is not None else ""
        with open(self.event_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([t, event_type, details, details_json])