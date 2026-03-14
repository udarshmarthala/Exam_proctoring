"""Head pose estimation (yaw/pitch/roll) using solvePnP on MediaPipe landmarks."""
from __future__ import annotations

import cv2
import numpy as np

from monitoring.config import MonitoringConfig
from monitoring.models import HeadPoseResult

# 3D model points of a generic face (nose tip centered at origin)
MODEL_POINTS = np.array([
    [0.0, 0.0, 0.0],          # Nose tip (landmark 1)
    [0.0, -63.6, -12.5],      # Chin (landmark 152)
    [-43.3, 32.7, -26.0],     # Left eye outer corner (landmark 33)
    [43.3, 32.7, -26.0],      # Right eye outer corner (landmark 263)
    [-28.9, -28.9, -24.1],    # Left mouth corner (landmark 61)
    [28.9, -28.9, -24.1],     # Right mouth corner (landmark 291)
], dtype=np.float64)

# Corresponding MediaPipe landmark indices
FACE_LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]


class HeadPoseEstimator:
    def __init__(self, config: MonitoringConfig):
        self.config = config

    def estimate_pose(self, landmarks, frame_shape: tuple) -> HeadPoseResult:
        """Estimate head pose from MediaPipe face landmarks.

        Args:
            landmarks: MediaPipe NormalizedLandmarkList for one face.
            frame_shape: (height, width, channels) of the frame.

        Returns:
            HeadPoseResult with yaw, pitch, roll in degrees.
        """
        h, w = frame_shape[:2]
        lm = landmarks.landmark

        # Extract 2D image points (denormalize from 0-1 to pixel coords)
        image_points = np.array(
            [[lm[i].x * w, lm[i].y * h] for i in FACE_LANDMARK_INDICES],
            dtype=np.float64,
        )

        # Approximate camera matrix
        focal_length = w
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return HeadPoseResult()

        # Convert rotation vector to euler angles
        rmat, _ = cv2.Rodrigues(rvec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        yaw = float(angles[1])
        pitch = float(angles[0])
        roll = float(angles[2])

        return HeadPoseResult(
            yaw=round(yaw, 1),
            pitch=round(pitch, 1),
            roll=round(roll, 1),
        )

    def is_turned_away(self, pose: HeadPoseResult) -> bool:
        return (
            abs(pose.yaw) > self.config.head_yaw_threshold
            or abs(pose.pitch) > self.config.head_pitch_threshold
        )

    def yaw_exceeds_warn(self, pose: HeadPoseResult) -> bool:
        return abs(pose.yaw) > self.config.head_yaw_warn_deg

    def yaw_exceeds_alert(self, pose: HeadPoseResult) -> bool:
        return abs(pose.yaw) > self.config.head_yaw_alert_deg

    def pitch_down_exceeds_warn(self, pose: HeadPoseResult) -> bool:
        return pose.pitch > self.config.head_pitch_down_warn_deg

    def pitch_up_exceeds_warn(self, pose: HeadPoseResult) -> bool:
        return pose.pitch < -self.config.head_pitch_up_warn_deg
