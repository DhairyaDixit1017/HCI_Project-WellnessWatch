from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from .config import AppConfig


@dataclass
class GazeFrameData:
    frame_bgr: np.ndarray
    face_detected: bool
    ear_left: Optional[float]
    ear_right: Optional[float]
    ear_avg: Optional[float]
    eyes_closed: bool
    head_yaw_deg: Optional[float]
    head_pitch_deg: Optional[float]
    gaze_offset_x: Optional[float] = None


class GazeTracker:

    def __init__(self, config: AppConfig):
        self.config = config
        self.cap = cv2.VideoCapture(config.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def release(self):
        self.cap.release()
        self.face_mesh.close()

    def read(self) -> Optional[GazeFrameData]:
        ok, frame = self.cap.read()
        if not ok:
            return None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        h, w, _ = frame.shape

        face_detected = False
        ear_left = ear_right = ear_avg = None
        yaw = pitch = None
        eyes_closed = False
        gaze_offset_x = None

        if results.multi_face_landmarks:
            face_detected = True
            face_landmarks = results.multi_face_landmarks[0]
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]

            ear_left = self._compute_eye_ear(pts, left=True)
            ear_right = self._compute_eye_ear(pts, left=False)

            if ear_left is not None and ear_right is not None:
                ear_avg = (ear_left + ear_right) / 2.0
                eyes_closed = ear_avg < self.config.drowsy_ear_threshold
            else:
                ear_avg = None

            yaw, pitch = self._estimate_head_orientation(pts, w, h)
            gaze_offset_x = self._compute_gaze_offset(pts)

        return GazeFrameData(
            frame_bgr=frame,
            face_detected=face_detected,
            ear_left=ear_left,
            ear_right=ear_right,
            ear_avg=ear_avg,
            eyes_closed=eyes_closed,
            head_yaw_deg=yaw,
            head_pitch_deg=pitch,
            gaze_offset_x=gaze_offset_x,
        )

    @staticmethod
    def _eye_indices(left: bool = True) -> Dict[str, int]:
        if left:
            return {
                "p1": 33,
                "p2": 160,
                "p3": 158,
                "p4": 133,
                "p5": 153,
                "p6": 144,
            }
        else:
            return {
                "p1": 263,
                "p2": 387,
                "p3": 385,
                "p4": 362,
                "p5": 380,
                "p6": 373,
            }

    def _compute_eye_ear(
        self, pts: Tuple[Tuple[int, int], ...], left: bool = True
    ) -> Optional[float]:
        idx = self._eye_indices(left)
        try:
            p1 = np.array(pts[idx["p1"]], dtype=np.float32)
            p2 = np.array(pts[idx["p2"]], dtype=np.float32)
            p3 = np.array(pts[idx["p3"]], dtype=np.float32)
            p4 = np.array(pts[idx["p4"]], dtype=np.float32)
            p5 = np.array(pts[idx["p5"]], dtype=np.float32)
            p6 = np.array(pts[idx["p6"]], dtype=np.float32)
        except (KeyError, IndexError):
            return None

        num = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
        den = 2.0 * np.linalg.norm(p1 - p4)
        if den <= 1e-6:
            return None
        return float(num / den)

    def _compute_gaze_offset(
        self, pts: Tuple[Tuple[int, int], ...]
    ) -> Optional[float]:
        def eye_offset(outer_idx: int, inner_idx: int, iris_idx: int) -> Optional[float]:
            try:
                outer = np.array(pts[outer_idx], dtype=np.float32)
                inner = np.array(pts[inner_idx], dtype=np.float32)
                iris = np.array(pts[iris_idx], dtype=np.float32)
            except IndexError:
                return None

            eye_width = np.linalg.norm(outer - inner)
            if eye_width < 1e-6:
                return None

            center = (outer + inner) / 2.0
            offset_x = float((iris[0] - center[0]) / eye_width)
            return offset_x

        left_off = eye_offset(33, 133, 468)
        right_off = eye_offset(263, 362, 473)

        offsets = [o for o in (left_off, right_off) if o is not None]
        if not offsets:
            return None
        return float(np.mean(offsets))

    def _estimate_head_orientation(
        self, pts: Tuple[Tuple[int, int], ...], w: int, h: int
    ) -> Tuple[Optional[float], Optional[float]]:
        try:
            nose_idx = 1
            left_eye_outer_idx = 33
            right_eye_outer_idx = 263
            mouth_left_idx = 61
            mouth_right_idx = 291
            chin_idx = 152

            image_points = np.array(
                [
                    pts[nose_idx],
                    pts[left_eye_outer_idx],
                    pts[right_eye_outer_idx],
                    pts[mouth_left_idx],
                    pts[mouth_right_idx],
                    pts[chin_idx],
                ],
                dtype="double",
            )
        except IndexError:
            return None, None

        model_points = np.array(
            [
                [0.0, 0.0, 0.0],
                [-30.0, 30.0, -30.0],
                [30.0, 30.0, -30.0],
                [-25.0, -30.0, -30.0],
                [25.0, -30.0, -30.0],
                [0.0, -60.0, -10.0],
            ],
            dtype="double",
        )

        focal_length = w
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array(
            [
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1],
            ],
            dtype="double",
        )
        dist_coeffs = np.zeros((4, 1))

        try:
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            return None, None

        if not success:
            return None, None

        rotation_mat, _ = cv2.Rodrigues(rotation_vector)

        sy = np.sqrt(rotation_mat[0, 0] ** 2 + rotation_mat[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            pitch = np.arctan2(-rotation_mat[2, 0], sy)
            yaw = np.arctan2(rotation_mat[1, 0], rotation_mat[0, 0])
        else:
            pitch = np.arctan2(-rotation_mat[2, 0], sy)
            yaw = 0.0

        yaw_deg = float(yaw * 180.0 / np.pi)
        pitch_deg = float(pitch * 180.0 / np.pi)
        return yaw_deg, pitch_deg