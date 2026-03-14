"""Eye gaze direction estimation using MediaPipe Face Mesh iris landmarks."""
from __future__ import annotations

import numpy as np

from monitoring.config import MonitoringConfig
from monitoring.models import GazeDirection, GazeResult

# MediaPipe Face Mesh landmark indices (refine_landmarks=True required for iris)
# Left eye
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
LEFT_IRIS_CENTER = 468

# Right eye
RIGHT_EYE_OUTER = 362
RIGHT_EYE_INNER = 263
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_IRIS_CENTER = 473


class GazeTracker:
    def __init__(self, config: MonitoringConfig):
        self.config = config

    def estimate_gaze(self, landmarks, frame_shape: tuple) -> GazeResult:
        """Estimate gaze direction from MediaPipe face landmarks.

        Args:
            landmarks: MediaPipe NormalizedLandmarkList for one face.
            frame_shape: (height, width, channels) of the frame.

        Returns:
            GazeResult with direction and ratios.
        """
        lm = landmarks.landmark

        # --- Horizontal ratio (average of both eyes) ---
        left_h = self._iris_horizontal_ratio(
            lm, LEFT_IRIS_CENTER, LEFT_EYE_INNER, LEFT_EYE_OUTER
        )
        right_h = self._iris_horizontal_ratio(
            lm, RIGHT_IRIS_CENTER, RIGHT_EYE_INNER, RIGHT_EYE_OUTER
        )
        h_ratio = (left_h + right_h) / 2.0

        # --- Vertical ratio (average of both eyes) ---
        left_v = self._iris_vertical_ratio(
            lm, LEFT_IRIS_CENTER, LEFT_EYE_TOP, LEFT_EYE_BOTTOM
        )
        right_v = self._iris_vertical_ratio(
            lm, RIGHT_IRIS_CENTER, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM
        )
        v_ratio = (left_v + right_v) / 2.0

        # --- Classify direction (horizontal takes precedence) ---
        direction = GazeDirection.CENTER
        if h_ratio < self.config.gaze_left_threshold:
            direction = GazeDirection.LEFT
        elif h_ratio > self.config.gaze_right_threshold:
            direction = GazeDirection.RIGHT
        elif v_ratio < self.config.gaze_up_threshold:
            direction = GazeDirection.UP
        elif v_ratio > self.config.gaze_down_threshold:
            direction = GazeDirection.DOWN

        ear = self._eye_aspect_ratio(lm)
        return GazeResult(
            direction=direction,
            horizontal_ratio=round(h_ratio, 3),
            vertical_ratio=round(v_ratio, 3),
            eye_aspect_ratio=round(ear, 3),
        )

    @staticmethod
    def _eye_aspect_ratio(lm) -> float:
        """Eye aspect ratio (EAR). Below ~0.2 = closed. Blink typically < 0.4s."""
        # Left: 33, 160, 158, 133, 153, 144; Right: 362, 385, 387, 263, 373, 380
        def _ear(pts):
            if len(pts) < 6:
                return 1.0
            v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
            v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
            h = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
            if h < 1e-6:
                return 1.0
            return (v1 + v2) / (2.0 * h)
        try:
            left = _ear([
                (lm[33].x, lm[33].y), (lm[160].x, lm[160].y), (lm[158].x, lm[158].y),
                (lm[133].x, lm[133].y), (lm[153].x, lm[153].y), (lm[144].x, lm[144].y),
            ])
            right = _ear([
                (lm[362].x, lm[362].y), (lm[385].x, lm[385].y), (lm[387].x, lm[387].y),
                (lm[263].x, lm[263].y), (lm[373].x, lm[373].y), (lm[380].x, lm[380].y),
            ])
            return (left + right) / 2.0
        except (IndexError, AttributeError):
            return 1.0

    @staticmethod
    def _iris_horizontal_ratio(lm, iris_idx: int, inner_idx: int, outer_idx: int) -> float:
        iris_x = lm[iris_idx].x
        inner_x = lm[inner_idx].x
        outer_x = lm[outer_idx].x
        denom = abs(outer_x - inner_x)
        if denom < 1e-6:
            return 0.5
        return (iris_x - min(inner_x, outer_x)) / denom

    @staticmethod
    def _iris_vertical_ratio(lm, iris_idx: int, top_idx: int, bottom_idx: int) -> float:
        iris_y = lm[iris_idx].y
        top_y = lm[top_idx].y
        bottom_y = lm[bottom_idx].y
        denom = abs(bottom_y - top_y)
        if denom < 1e-6:
            return 0.5
        return (iris_y - top_y) / denom
