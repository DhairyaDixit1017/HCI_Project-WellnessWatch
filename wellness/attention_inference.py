from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List

from tracker_core.config import AppConfig
from tracker_core.gaze_tracker import GazeFrameData
from .posture_module import PostureData


class SystemState(Enum):
    OK = auto()
    DISTRACTED = auto()
    DROWSY = auto()
    POSTURE_RISK = auto()


@dataclass
class StateOutput:
    state: SystemState
    reasons: List[str] = field(default_factory=list)


class AttentionInference:
    """
    Rule-based fusion of:
    - Head yaw (coarse orientation)
    - Gaze offset (iris vs eye center)
    - Face presence (no face = away from screen)
    - EAR-based drowsiness
    - Posture slouch angle
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self._distract_acc = 0.0
        self._gaze_distract_acc = 0.0
        self._no_face_acc = 0.0
        self._drowsy_acc = 0.0
        self._posture_acc = 0.0

    def reset(self):
        self._distract_acc = 0.0
        self._gaze_distract_acc = 0.0
        self._no_face_acc = 0.0
        self._drowsy_acc = 0.0
        self._posture_acc = 0.0

    def update(
        self,
        gaze: GazeFrameData,
        posture: PostureData,
        dt_seconds: float,
    ) -> StateOutput:
        reasons: List[str] = []
        state = SystemState.OK

        yaw_distracted = (
            gaze.head_yaw_deg is not None
            and abs(gaze.head_yaw_deg) > self.config.distraction_yaw_deg
        )
        if yaw_distracted:
            self._distract_acc += dt_seconds
        else:
            self._distract_acc = max(0.0, self._distract_acc - dt_seconds)

        gaze_off = gaze.gaze_offset_x
        gaze_distracted = (
            gaze_off is not None
            and abs(gaze_off) > self.config.gaze_offset_threshold
        )
        if gaze_distracted:
            self._gaze_distract_acc += dt_seconds
        else:
            self._gaze_distract_acc = max(0.0, self._gaze_distract_acc - dt_seconds)

        if not gaze.face_detected:
            self._no_face_acc += dt_seconds
        else:
            self._no_face_acc = max(0.0, self._no_face_acc - dt_seconds)

        distracted = (
            self._distract_acc >= self.config.distraction_min_seconds
            or self._gaze_distract_acc >= self.config.gaze_distraction_min_seconds
            or self._no_face_acc >= self.config.gaze_distraction_min_seconds
        )

        if distracted:
            state = SystemState.DISTRACTED
            if self._no_face_acc >= self.config.gaze_distraction_min_seconds:
                reasons.append("no_face_detected")
            else:
                if yaw_distracted:
                    reasons.append("head_yaw_large")
                if gaze_distracted:
                    reasons.append("gaze_offset_large")

        if gaze.eyes_closed:
            self._drowsy_acc += dt_seconds
        else:
            self._drowsy_acc = max(0.0, self._drowsy_acc - dt_seconds)

        if self._drowsy_acc >= self.config.drowsy_min_seconds:
            state = SystemState.DROWSY
            reasons = ["eyes_closed_long"]

        if posture.posture_state == "SLOUCH":
            self._posture_acc += dt_seconds
        else:
            self._posture_acc = max(0.0, self._posture_acc - dt_seconds)

        if self._posture_acc >= self.config.posture_min_seconds:
            state = SystemState.POSTURE_RISK
            reasons = ["slouching"]

        if not reasons:
            reasons.append("normal")

        return StateOutput(state=state, reasons=reasons)