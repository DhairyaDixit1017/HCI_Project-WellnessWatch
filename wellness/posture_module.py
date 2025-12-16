from dataclasses import dataclass
from collections import deque
from typing import Optional, Deque, Tuple

import cv2
import mediapipe as mp
import numpy as np

from tracker_core.config import AppConfig


@dataclass
class PostureData:
    posture_state: str
    torso_tilt_deg: Optional[float]
    lean_score: Optional[float]
    forward_head_score: Optional[float] 

class PostureModule:

    def __init__(self, config: AppConfig):
        self.config = config
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.state_window: Deque[str] = deque(maxlen=9)
        self.good_streak = 0

        self.visibility_thresh = 0.35
        self.lean_threshold = 0.12
        self.torso_tilt_threshold_deg = 12.0
        self.forward_head_z_threshold = 0.05

    def close(self):
        self.pose.close()

    def update(self, frame_bgr) -> PostureData:
        if not self.config.enable_posture:
            return PostureData("DISABLED", None, None, None)

        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        if not results.pose_landmarks:
            return self._smoothed(PostureData("UNKNOWN", None, None, None))

        lm = results.pose_landmarks.landmark
        P = self.mp_pose.PoseLandmark

        nose = self._get_lm(lm[P.NOSE], w, h)
        L_sh = self._get_lm(lm[P.LEFT_SHOULDER], w, h)
        R_sh = self._get_lm(lm[P.RIGHT_SHOULDER], w, h)
        L_hip = self._get_lm(lm[P.LEFT_HIP], w, h)
        R_hip = self._get_lm(lm[P.RIGHT_HIP], w, h)

        if L_sh is None or R_sh is None:
            return self._smoothed(PostureData("UNKNOWN", None, None, None))

        shoulder_mid_xy = (L_sh[0] + R_sh[0]) / 2.0
        shoulder_mid_z = (L_sh[2] + R_sh[2]) / 2.0

        shoulder_dist = float(np.linalg.norm(L_sh[0] - R_sh[0])) + 1e-6
        lean_score = float((R_sh[0][1] - L_sh[0][1]) / shoulder_dist)

        torso_tilt_deg = None
        if L_hip is not None and R_hip is not None:
            hip_mid_xy = (L_hip[0] + R_hip[0]) / 2.0
            torso_vec = shoulder_mid_xy - hip_mid_xy
            vertical = np.array([0.0, -1.0], dtype=np.float32)
            torso_unit = torso_vec / (np.linalg.norm(torso_vec) + 1e-6)
            cosang = float(np.clip(np.dot(torso_unit, vertical), -1.0, 1.0))
            torso_tilt_deg = float(np.degrees(np.arccos(cosang)))

        forward_head_score = None
        if nose is not None:
            forward_head_score = float(shoulder_mid_z - nose[2])

        state = "GOOD"

        if forward_head_score is not None and forward_head_score > self.forward_head_z_threshold:
            state = "FORWARD_HEAD"

        if torso_tilt_deg is not None and torso_tilt_deg > self.torso_tilt_threshold_deg:
            state = "SLOUCH"

        if state == "GOOD":
            if lean_score > self.lean_threshold:
                state = "LEAN_RIGHT"
            elif lean_score < -self.lean_threshold:
                state = "LEAN_LEFT"

        return self._smoothed(PostureData(state, torso_tilt_deg, lean_score, forward_head_score))

    def _get_lm(self, lmk, w: int, h: int) -> Optional[Tuple[np.ndarray, float, float]]:

        if getattr(lmk, "visibility", 1.0) < self.visibility_thresh:
            return None
        xy = np.array([lmk.x * w, lmk.y * h], dtype=np.float32)
        return (xy, float(lmk.z), float(getattr(lmk, "visibility", 1.0)))

    def _smoothed(self, data: PostureData) -> PostureData:
        if data.posture_state == "GOOD":
            self.good_streak += 1
        else:
            self.good_streak = 0

        if self.good_streak >= 3:
            self.state_window.clear()
            self.state_window.append("GOOD")
            return data

        self.state_window.append(data.posture_state)

        counts = {}
        for s in self.state_window:
            counts[s] = counts.get(s, 0) + 1
        majority = max(counts.items(), key=lambda kv: kv[1])[0]

        return PostureData(majority, data.torso_tilt_deg, data.lean_score, data.forward_head_score)
