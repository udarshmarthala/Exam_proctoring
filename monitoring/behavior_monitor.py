"""Exam proctoring: gaze, head pose, face detection with L1/L2/L3 alerts and escalation."""
from __future__ import annotations

import os
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

from monitoring.config import MonitoringConfig
from monitoring.gaze_tracker import GazeTracker
from monitoring.head_pose import HeadPoseEstimator
from monitoring.models import (
    BehaviorFlag,
    GazeDirection,
    GazeResult,
    HeadPoseResult,
    MonitoringFrame,
    ProctoringEventType,
)

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")


class _LandmarkAdapter:
    def __init__(self, task_landmarks):
        self.landmark = task_landmarks


def _gaze_to_vector(g: Optional[GazeResult]) -> Optional[List[float]]:
    if g is None:
        return None
    return [g.horizontal_ratio, g.vertical_ratio]


def _avg_pose(points: Deque[Tuple[float, HeadPoseResult]]) -> HeadPoseResult:
    if not points:
        return HeadPoseResult()
    n = len(points)
    yaw = sum(p[1].yaw for p in points) / n
    pitch = sum(p[1].pitch for p in points) / n
    roll = sum(p[1].roll for p in points) / n
    return HeadPoseResult(yaw=round(yaw, 1), pitch=round(pitch, 1), roll=round(roll, 1))


def _avg_gaze(points: Deque[Tuple[float, GazeResult]]) -> Optional[GazeResult]:
    if not points:
        return None
    g0 = points[0][1]
    n = len(points)
    h = sum(p[1].horizontal_ratio for p in points) / n
    v = sum(p[1].vertical_ratio for p in points) / n
    ear = sum(p[1].eye_aspect_ratio for p in points) / n
    direction = g0.direction  # keep last direction for display
    return GazeResult(
        direction=direction,
        horizontal_ratio=round(h, 3),
        vertical_ratio=round(v, 3),
        eye_aspect_ratio=round(ear, 3),
    )


class BehaviorMonitor:
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self.gaze_tracker = GazeTracker(self.config)
        self.head_pose_estimator = HeadPoseEstimator(self.config)

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_faces=self.config.max_num_faces,
            min_face_detection_confidence=self.config.min_detection_confidence,
            min_face_presence_confidence=self.config.min_tracking_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)

        self._flag_timers: Dict[BehaviorFlag, float] = {}
        self._session_start: Optional[float] = None
        self._smooth_window_s = self.config.smoothing_window_ms / 1000.0
        self._pose_buffer: Deque[Tuple[float, HeadPoseResult]] = deque(maxlen=32)
        self._gaze_buffer: Deque[Tuple[float, GazeResult]] = deque(maxlen=32)
        self._yaw_warn_start: Optional[float] = None
        self._pitch_down_start: Optional[float] = None
        self._pitch_up_start: Optional[float] = None
        self._gaze_off_start: Optional[float] = None
        self._eye_closed_start: Optional[float] = None
        self._face_absent_start: Optional[float] = None
        self._l1_event_times: List[float] = []
        self._l2_count: int = 0
        self._l3_escalation_fired: bool = False
        self._sneeze_suppress_until: float = 0.0
        self._last_eye_close_time: Optional[float] = None
        self._last_head_deviation_time: Optional[float] = None

    def _in_grace(self, now: float) -> bool:
        if self._session_start is None:
            self._session_start = now
        return (now - self._session_start) < self.config.grace_period_s

    def _prune_l1_events(self, now: float) -> None:
        expire = now - self.config.escalation_l1_expire_s
        window = now - self.config.escalation_l1_window_s
        self._l1_event_times = [t for t in self._l1_event_times if t > expire and t > window]

    def _add_alert(
        self,
        alerts: List[dict],
        level: int,
        event_type: str,
        message: str,
        now: float,
        duration_ms: float = 0,
        head_pose: Optional[HeadPoseResult] = None,
        gaze: Optional[GazeResult] = None,
        confidence: float = 0.9,
    ) -> None:
        alerts.append({
            "level": level,
            "event_type": event_type,
            "message": message,
            "duration_ms": duration_ms,
            "head_yaw": head_pose.yaw if head_pose else None,
            "head_pitch": head_pose.pitch if head_pose else None,
            "gaze_vector": _gaze_to_vector(gaze),
            "confidence_score": confidence,
        })
        if level == 1:
            self._l1_event_times.append(now)
            self._prune_l1_events(now)
        elif level == 2:
            self._l2_count += 1

    def process_frame(self, frame: np.ndarray) -> MonitoringFrame:
        now = time.time()
        rgb = frame[:, :, ::-1].copy()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._landmarker.detect(mp_image)

        face_count = 0
        gaze: Optional[GazeResult] = None
        head_pose: Optional[HeadPoseResult] = None
        flags: List[BehaviorFlag] = []
        alerts: List[dict] = []
        confidence = 0.0

        if results.face_landmarks:
            face_count = len(results.face_landmarks)
            primary = _LandmarkAdapter(results.face_landmarks[0])
            gaze = self.gaze_tracker.estimate_gaze(primary, frame.shape)
            head_pose = self.head_pose_estimator.estimate_pose(primary, frame.shape)
            confidence = 0.9

            self._pose_buffer.append((now, head_pose))
            self._gaze_buffer.append((now, gaze))
            cutoff = now - self._smooth_window_s
            while self._pose_buffer and self._pose_buffer[0][0] < cutoff:
                self._pose_buffer.popleft()
            while self._gaze_buffer and self._gaze_buffer[0][0] < cutoff:
                self._gaze_buffer.popleft()

            smooth_pose = _avg_pose(self._pose_buffer) if self._pose_buffer else head_pose
            smooth_gaze = _avg_gaze(self._gaze_buffer) if self._gaze_buffer else gaze
            if smooth_gaze is None:
                smooth_gaze = gaze

            if confidence < self.config.confidence_threshold:
                smooth_pose = head_pose
                smooth_gaze = gaze or smooth_gaze

            # Use smoothed for thresholds
            head_pose = smooth_pose
            gaze = smooth_gaze or gaze

            # ----- Multiple faces → L2 immediately -----
            if face_count > 1 and not self._in_grace(now):
                flags.append(BehaviorFlag.MULTIPLE_FACES)
                self._add_alert(alerts, 2, ProctoringEventType.multiple_faces.value, "Multiple faces detected", now)

            # ----- Yaw > 30° → L2 even brief -----
            if not self.config.disable_yaw and not self._in_grace(now):
                if self.head_pose_estimator.yaw_exceeds_alert(head_pose):
                    flags.append(BehaviorFlag.HEAD_TURNED)
                    self._add_alert(
                        alerts, 2, ProctoringEventType.head_turn.value,
                        "Head turned too far (yaw > 30°)", now,
                        head_pose=head_pose, gaze=gaze, confidence=confidence,
                    )
                    self._yaw_warn_start = None
                elif self.head_pose_estimator.yaw_exceeds_warn(head_pose):
                    self._last_head_deviation_time = now
                    if self._yaw_warn_start is None:
                        self._yaw_warn_start = now
                    elif (now - self._yaw_warn_start) >= self.config.head_yaw_warn_sustained_s:
                        flags.append(BehaviorFlag.HEAD_TURNED)
                        self._add_alert(
                            alerts, 1, ProctoringEventType.head_turn.value,
                            "Please look at your screen", now,
                            duration_ms=(now - self._yaw_warn_start) * 1000,
                            head_pose=head_pose, gaze=gaze, confidence=confidence,
                        )
                        self._yaw_warn_start = None
                else:
                    self._yaw_warn_start = None

            # ----- Pitch down > 25° sustained 2s → L1 -----
            if not self.config.disable_pitch and not self._in_grace(now):
                if self.head_pose_estimator.pitch_down_exceeds_warn(head_pose):
                    if self._pitch_down_start is None:
                        self._pitch_down_start = now
                    elif (now - self._pitch_down_start) >= self.config.head_pitch_down_sustained_s:
                        self._add_alert(
                            alerts, 1, ProctoringEventType.head_turn.value,
                            "Please look at your screen", now,
                            duration_ms=(now - self._pitch_down_start) * 1000,
                            head_pose=head_pose, gaze=gaze, confidence=confidence,
                        )
                        self._pitch_down_start = None
                else:
                    self._pitch_down_start = None
                if self.head_pose_estimator.pitch_up_exceeds_warn(head_pose):
                    if self._pitch_up_start is None:
                        self._pitch_up_start = now
                    elif (now - self._pitch_up_start) >= self.config.head_pitch_up_sustained_s:
                        self._add_alert(
                            alerts, 1, ProctoringEventType.head_turn.value,
                            "Please look at your screen", now,
                            duration_ms=(now - self._pitch_up_start) * 1000,
                            head_pose=head_pose, gaze=gaze, confidence=confidence,
                        )
                        self._pitch_up_start = None
                else:
                    self._pitch_up_start = None

            # ----- Gaze off-center > 15° sustained 1.5s → L1 -----
            gaze_off = gaze and gaze.direction != GazeDirection.CENTER
            if gaze_off and not self._in_grace(now):
                if self._gaze_off_start is None:
                    self._gaze_off_start = now
                elif (now - self._gaze_off_start) >= self.config.gaze_sustained_s:
                    flags.append(BehaviorFlag.LOOKING_AWAY)
                    self._add_alert(
                        alerts, 1, ProctoringEventType.gaze_deviation.value,
                        "Please look at your screen", now,
                        duration_ms=(now - self._gaze_off_start) * 1000,
                        head_pose=head_pose, gaze=gaze, confidence=confidence,
                    )
                    self._gaze_off_start = None
            else:
                self._gaze_off_start = None

            # ----- Eye closure > 2s (not blink < 400ms) → L1 -----
            ear = gaze.eye_aspect_ratio if gaze else 1.0
            closed = ear < self.config.eye_aspect_ratio_closed
            if closed:
                self._last_eye_close_time = self._last_eye_close_time or now
            if closed and not self._in_grace(now) and now > self._sneeze_suppress_until:
                if self._eye_closed_start is None:
                    self._eye_closed_start = now
                else:
                    closed_duration = now - self._eye_closed_start
                    if closed_duration >= self.config.eye_closure_s:
                        self._add_alert(
                            alerts, 1, ProctoringEventType.eye_closure.value,
                            "Eyes closed for 2+ seconds", now,
                            duration_ms=closed_duration * 1000,
                            head_pose=head_pose, gaze=gaze, confidence=confidence,
                        )
                        self._eye_closed_start = None
            else:
                if self._eye_closed_start is not None:
                    closed_duration = now - self._eye_closed_start
                    if closed_duration <= self.config.blink_max_s:
                        pass  # blink, ignore
                    elif (self._last_head_deviation_time and
                          (now - self._last_head_deviation_time) <= self.config.sneeze_resolve_s):
                        self._sneeze_suppress_until = now + self.config.sneeze_resolve_s
                self._eye_closed_start = None
                if not closed:
                    self._last_eye_close_time = None

            self._face_absent_start = None
            self._flag_timers.pop(BehaviorFlag.ABSENT, None)

        else:
            self._pose_buffer.clear()
            self._gaze_buffer.clear()
            if self._face_absent_start is None:
                self._face_absent_start = now
            absent_duration = now - self._face_absent_start
            if not self._in_grace(now):
                if absent_duration >= self.config.face_absent_l3_s:
                    flags.append(BehaviorFlag.ABSENT)
                    self._add_alert(
                        alerts, 3, ProctoringEventType.face_absent.value,
                        "No face detected for 10+ seconds", now,
                        duration_ms=absent_duration * 1000,
                    )
                elif absent_duration >= self.config.face_absent_l2_s:
                    flags.append(BehaviorFlag.ABSENT)
                    self._add_alert(
                        alerts, 2, ProctoringEventType.face_absent.value,
                        "No face detected for 3+ seconds", now,
                        duration_ms=absent_duration * 1000,
                    )

        # Escalation: 3 L1 in 5 min → L2
        self._prune_l1_events(now)
        if len(self._l1_event_times) >= self.config.escalation_l1_count and not self._in_grace(now):
            self._l1_event_times.clear()
            self._add_alert(alerts, 2, ProctoringEventType.gaze_deviation.value,
                           "Auto-escalated: multiple Level 1 warnings in 5 minutes", now)
        if (not self._l3_escalation_fired and self._l2_count >= self.config.escalation_l2_count
                and not self._in_grace(now)):
            self._l3_escalation_fired = True
            self._add_alert(alerts, 3, ProctoringEventType.gaze_deviation.value,
                            "Auto-escalated: multiple Level 2 alerts", now)

        if not flags:
            flags.append(BehaviorFlag.NORMAL)

        # Low light: mean brightness on small center crop
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        h, w = gray.shape[:2]
        cx, cy = w // 4, h // 4
        crop = gray[cy : cy + h // 2, cx : cx + w // 2]
        low_light = float(np.mean(crop)) < 60.0

        return MonitoringFrame(
            timestamp=now,
            gaze=gaze,
            head_pose=head_pose,
            face_count=face_count,
            flags=flags,
            alerts=alerts,
            confidence=confidence,
            low_light=low_light,
        )

    def reset(self) -> None:
        self._flag_timers.clear()
        self._session_start = None
        self._pose_buffer.clear()
        self._gaze_buffer.clear()
        self._yaw_warn_start = None
        self._pitch_down_start = None
        self._pitch_up_start = None
        self._gaze_off_start = None
        self._eye_closed_start = None
        self._face_absent_start = None
        self._l1_event_times.clear()
        self._l2_count = 0
        self._l3_escalation_fired = False
        self._sneeze_suppress_until = 0.0
        self._last_eye_close_time = None
        self._last_head_deviation_time = None

    def release(self) -> None:
        self._landmarker.close()
