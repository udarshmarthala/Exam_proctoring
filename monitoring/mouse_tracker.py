"""Mouse movement and click tracking for exam proctoring.

Detects:
- Mouse leaving the exam window/browser
- Erratic or unusual movement patterns (e.g. off-screen frequently, very fast jumps)
- Unusual click patterns (bot-like rapid clicks) or prolonged inactivity
"""
from __future__ import annotations

import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

from monitoring.config import MonitoringConfig
from monitoring.models import ProctoringEventType


def _speed_px_per_sec(
    x1: float, y1: float, t1: float,
    x2: float, y2: float, t2: float,
) -> float:
    if t2 <= t1:
        return 0.0
    dx, dy = x2 - x1, y2 - y1
    return (dx * dx + dy * dy) ** 0.5 / (t2 - t1)


class MouseTracker:
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self._last_movement_time: Optional[float] = None
        self._last_inside_window: Optional[bool] = None
        self._leave_window_start: Optional[float] = None
        self._movement_history: Deque[Tuple[float, float, float, bool]] = deque(maxlen=64)  # (t, x, y, inside)
        self._click_times: Deque[float] = deque(maxlen=32)
        self._session_start: Optional[float] = None

    def _ensure_session(self, now: float) -> None:
        if self._session_start is None:
            self._session_start = now

    def process_batch(
        self,
        movements: List[dict],
        clicks: List[dict],
        last_snapshot: Optional[dict],
        now: Optional[float] = None,
    ) -> List[dict]:
        """Process a batch of mouse movements and clicks from the client.
        Returns list of alerts (same format as behavior_monitor alerts)."""
        now = now or time.time()
        self._ensure_session(now)
        alerts: List[dict] = []

        # --- Movement history and inside_window ---
        for m in movements:
            t = m.get("t", now)
            x = m.get("x", 0)
            y = m.get("y", 0)
            inside = m.get("inside", True)
            self._movement_history.append((t, x, y, inside))
            self._last_movement_time = t
            self._last_inside_window = inside

        if last_snapshot:
            t = last_snapshot.get("timestamp", now)
            inside = last_snapshot.get("inside_window", True)
            self._last_inside_window = inside
            if not inside:
                if self._leave_window_start is None:
                    self._leave_window_start = t
                elif (t - self._leave_window_start) >= self.config.mouse_leave_alert_s:
                    alerts.append({
                        "level": 2,
                        "event_type": ProctoringEventType.mouse_leave_window.value,
                        "message": "Mouse left the exam window",
                        "duration_ms": (t - self._leave_window_start) * 1000,
                        "confidence_score": 0.95,
                    })
            else:
                self._leave_window_start = None
        else:
            # No snapshot; if we had leave started and no recent inside, keep it
            if self._leave_window_start is not None and (now - self._leave_window_start) >= self.config.mouse_leave_alert_s:
                alerts.append({
                    "level": 2,
                    "event_type": ProctoringEventType.mouse_leave_window.value,
                    "message": "Mouse left the exam window",
                    "duration_ms": (now - self._leave_window_start) * 1000,
                    "confidence_score": 0.9,
                })
            # If we got movements with inside=False, we already updated _leave_window_start above

        # --- Erratic movement: high speed bursts ---
        window_s = self.config.mouse_erratic_window_s
        threshold_speed = self.config.mouse_erratic_speed_threshold
        burst_required = self.config.mouse_erratic_burst_count
        history = list(self._movement_history)
        if len(history) >= 2:
            fast_count = 0
            for i in range(1, len(history)):
                t1, x1, y1, _ = history[i - 1]
                t2, x2, y2, _ = history[i]
                if t2 - t1 <= 0:
                    continue
                if t2 < now - window_s:
                    continue
                speed = _speed_px_per_sec(x1, y1, t1, x2, y2, t2)
                if speed >= threshold_speed:
                    fast_count += 1
            if fast_count >= burst_required:
                alerts.append({
                    "level": 1,
                    "event_type": ProctoringEventType.mouse_erratic.value,
                    "message": "Erratic or unusual mouse movement detected",
                    "duration_ms": 0,
                    "confidence_score": 0.85,
                })

        # --- Inactivity: no movement for a long time ---
        if self._last_movement_time is not None:
            inactive_s = now - self._last_movement_time
            if inactive_s >= self.config.mouse_inactivity_s:
                alerts.append({
                    "level": 1,
                    "event_type": ProctoringEventType.mouse_inactivity.value,
                    "message": "Prolonged mouse inactivity",
                    "duration_ms": inactive_s * 1000,
                    "confidence_score": 0.9,
                })

        # --- Unusual click pattern: very fast repeated clicks ---
        for c in clicks:
            self._click_times.append(c.get("t", now))
        interval_ms = self.config.mouse_unusual_click_interval_ms
        count_required = self.config.mouse_unusual_click_count
        clicks_sorted = sorted(self._click_times)
        for i in range(len(clicks_sorted) - count_required + 1):
            window = clicks_sorted[i : i + count_required]
            if (window[-1] - window[0]) * 1000 <= interval_ms * count_required:
                alerts.append({
                    "level": 1,
                    "event_type": ProctoringEventType.mouse_unusual_clicks.value,
                    "message": "Unusual rapid click pattern detected",
                    "duration_ms": 0,
                    "confidence_score": 0.85,
                })
                break

        return alerts

    def reset(self) -> None:
        self._last_movement_time = None
        self._last_inside_window = None
        self._leave_window_start = None
        self._movement_history.clear()
        self._click_times.clear()
        self._session_start = None
