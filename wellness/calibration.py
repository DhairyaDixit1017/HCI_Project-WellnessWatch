import time
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import cv2

from tracker_core.config import AppConfig
from tracker_core.gaze_tracker import GazeTracker, GazeFrameData
from wellness.posture_module import PostureModule, PostureData


@dataclass
class BaselineConfig:
    ear_avg: Optional[float]
    head_yaw_deg: Optional[float]
    head_pitch_deg: Optional[float]
    torso_tilt_deg: Optional[float]
    forward_head_score: Optional[float]
    lean_score: Optional[float]


class CalibrationManager:

    def __init__(self, config: AppConfig):
        self.config = config

    def run(
        self,
        gaze_tracker: GazeTracker,
        posture_module: PostureModule,
        seconds: float = 3.0,
    ) -> BaselineConfig:
        end_time = time.time() + seconds

        ear_vals: List[float] = []
        yaw_vals: List[float] = []
        pitch_vals: List[float] = []

        torso_tilt_vals: List[float] = []
        forward_head_vals: List[float] = []
        lean_vals: List[float] = []

        while time.time() < end_time:
            frame_data: Optional[GazeFrameData] = gaze_tracker.read()
            if frame_data is None:
                break

            posture_data: PostureData = posture_module.update(frame_data.frame_bgr)

            if frame_data.ear_avg is not None:
                ear_vals.append(frame_data.ear_avg)
            if frame_data.head_yaw_deg is not None:
                yaw_vals.append(frame_data.head_yaw_deg)
            if frame_data.head_pitch_deg is not None:
                pitch_vals.append(frame_data.head_pitch_deg)

            if posture_data.torso_tilt_deg is not None:
                torso_tilt_vals.append(posture_data.torso_tilt_deg)
            if posture_data.forward_head_score is not None:
                forward_head_vals.append(posture_data.forward_head_score)
            if posture_data.lean_score is not None:
                lean_vals.append(posture_data.lean_score)

            cv2.putText(
                frame_data.frame_bgr,
                "Calibrating: sit upright and look at the screen...",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("WellnessWatch - Calibration", frame_data.frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyWindow("WellnessWatch - Calibration")

        baseline = BaselineConfig(
            ear_avg=_median_or_none(ear_vals),
            head_yaw_deg=_median_or_none(yaw_vals),
            head_pitch_deg=_median_or_none(pitch_vals),
            torso_tilt_deg=_median_or_none(torso_tilt_vals),
            forward_head_score=_median_or_none(forward_head_vals),
            lean_score=_median_or_none(lean_vals),
        )
        print("[Calibration] Baseline:", baseline)
        return baseline


def _median_or_none(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    return float(np.median(np.array(vals, dtype=np.float32)))