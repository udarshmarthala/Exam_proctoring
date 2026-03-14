#!/usr/bin/env python3
"""
Standalone webcam test for the monitoring module.

Run from the project root:
    python -m monitoring.test_standalone

Press 'q' to quit.
"""
from __future__ import annotations

import sys
import os

# Ensure project root is on path so `monitoring` package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from monitoring.behavior_monitor import BehaviorMonitor
from monitoring.config import MonitoringConfig
from monitoring.models import BehaviorFlag, GazeDirection

# Colors (BGR)
GREEN = (0, 255, 100)
RED = (0, 0, 255)
AMBER = (0, 180, 255)
WHITE = (255, 255, 255)


def draw_text(frame: np.ndarray, text: str, color: tuple, y: int = 30):
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main():
    config = MonitoringConfig()
    monitor = BehaviorMonitor(config)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        sys.exit(1)

    print("=" * 50)
    print("  Monitoring Module — Standalone Test")
    print("  Press 'q' to quit")
    print("=" * 50)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror for natural interaction
        result = monitor.process_frame(frame)
        frame_count += 1

        # --- Draw gaze info ---
        if result.gaze:
            gaze_color = GREEN if result.gaze.direction == GazeDirection.CENTER else AMBER
            draw_text(frame, f"Gaze: {result.gaze.direction.value.upper()}", gaze_color, 30)
            draw_text(
                frame,
                f"H: {result.gaze.horizontal_ratio:.2f}  V: {result.gaze.vertical_ratio:.2f}",
                WHITE, 55,
            )

        # --- Draw head pose ---
        if result.head_pose:
            draw_text(
                frame,
                f"Yaw: {result.head_pose.yaw:.0f}  Pitch: {result.head_pose.pitch:.0f}  Roll: {result.head_pose.roll:.0f}",
                WHITE, 80,
            )

        # --- Draw face count ---
        fc_color = GREEN if result.face_count == 1 else RED
        draw_text(frame, f"Faces: {result.face_count}", fc_color, 105)

        # --- Draw flags ---
        flag_y = frame.shape[0] - 20
        for flag in result.flags:
            color = GREEN if flag == BehaviorFlag.NORMAL else RED
            cv2.putText(
                frame, f"[{flag.value.upper()}]",
                (10, flag_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
            )
            flag_y -= 30

        # --- Print to console every 30 frames ---
        if frame_count % 30 == 0:
            flag_names = [f.value for f in result.flags]
            gaze_dir = result.gaze.direction.value if result.gaze else "N/A"
            yaw = result.head_pose.yaw if result.head_pose else 0
            print(
                f"[Frame {frame_count:>5}] "
                f"Faces={result.face_count}  "
                f"Gaze={gaze_dir:<7}  "
                f"Yaw={yaw:>5.1f}  "
                f"Flags={flag_names}"
            )

        cv2.imshow("Monitoring Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    monitor.release()
    cap.release()
    cv2.destroyAllWindows()
    print("\nTest complete.")


if __name__ == "__main__":
    main()
