"""Keyboard activity tracking for exam proctoring.

- Keystroke logging: records what keys are typed (key codes/names, no sensitive text).
- Keystroke dynamics: rhythm and timing between keystrokes (for identity verification).
- Forbidden shortcuts: Ctrl+C, Alt+Tab, Win key, etc.
- Typing pauses that suggest copy-pasting (long gap after last key).
"""
from __future__ import annotations

import time
from collections import deque
from typing import Deque, List, Optional, Set, Tuple

from monitoring.config import MonitoringConfig
from monitoring.models import ProctoringEventType


def _shortcut_string(ctrl: bool, alt: bool, meta: bool, shift: bool, key: str) -> str:
    parts = []
    if ctrl:
        parts.append("Control")
    if alt:
        parts.append("Alt")
    if meta:
        parts.append("Meta")
    if shift:
        parts.append("Shift")
    if key and key not in ("Control", "Alt", "Meta", "Shift"):
        parts.append(key)
    return "+".join(parts)


class KeyboardTracker:
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self._forbidden: Set[str] = set(self._normalize_shortcuts(self.config.forbidden_shortcuts))
        self._last_key_time: Optional[float] = None
        self._intervals: Deque[float] = deque(maxlen=self.config.keystroke_dynamics_window)
        self._session_start: Optional[float] = None
        self._last_keystroke_log: List[dict] = []  # last N for audit (key/code only, no content)

    @staticmethod
    def _normalize_shortcuts(shortcuts: Tuple[str, ...]) -> List[str]:
        out = []
        for s in shortcuts:
            s = s.strip().lower().replace(" ", "")
            if s:
                out.append(s)
        return out

    def process_batch(
        self,
        key_events: List[dict],
        now: Optional[float] = None,
    ) -> Tuple[List[dict], List[dict]]:
        """Process a batch of key events from the client.
        Returns (alerts, keystroke_log_entries).
        keystroke_log_entries are for audit: [{key, code, keydown, timestamp}, ...] (no actual text)."""
        now = now or time.time()
        if self._session_start is None:
            self._session_start = now
        alerts: List[dict] = []
        log_entries: List[dict] = []

        for ev in key_events:
            t = ev.get("timestamp", now)
            key = (ev.get("key") or "").strip()
            code = (ev.get("code") or "").strip()
            keydown = ev.get("keydown", True)
            ctrl = ev.get("ctrl", False)
            alt = ev.get("alt", False)
            meta = ev.get("meta", False)
            shift = ev.get("shift", False)

            # Keystroke logging (audit: what key was pressed, not content)
            log_entries.append({
                "timestamp": t,
                "key": key,
                "code": code,
                "keydown": keydown,
                "modifiers": _shortcut_string(ctrl, alt, meta, shift, ""),
            })

            if not keydown:
                continue

            # Forbidden shortcut check: exact match only (avoids "t" matching "alt+tab", "n" matching "control")
            shortcut = _shortcut_string(ctrl, alt, meta, shift, key if key else code)
            shortcut_lower = shortcut.lower().replace(" ", "")
            if shortcut_lower in self._forbidden:
                alerts.append({
                    "level": 3,
                    "event_type": ProctoringEventType.forbidden_shortcut.value,
                    "message": f"Forbidden shortcut used: {shortcut}",
                    "duration_ms": 0,
                    "confidence_score": 1.0,
                    "shortcut": shortcut,
                })

            # Keystroke dynamics: interval since last key
            if self._last_key_time is not None:
                interval_ms = (t - self._last_key_time) * 1000
                self._intervals.append(interval_ms)
                # Typing pause suggesting copy-paste
                if interval_ms >= self.config.copy_paste_pause_ms:
                    alerts.append({
                        "level": 1,
                        "event_type": ProctoringEventType.copy_paste_suspected.value,
                        "message": "Long typing pause detected (possible copy-paste)",
                        "duration_ms": interval_ms,
                        "confidence_score": 0.75,
                    })
            self._last_key_time = t

        # Keyboard inactivity
        if self._last_key_time is not None:
            inactive_s = now - self._last_key_time
            if inactive_s >= self.config.keyboard_inactivity_s:
                alerts.append({
                    "level": 1,
                    "event_type": ProctoringEventType.keyboard_inactivity.value,
                    "message": "Prolonged keyboard inactivity",
                    "duration_ms": inactive_s * 1000,
                    "confidence_score": 0.9,
                })

        return alerts, log_entries

    def get_keystroke_dynamics_intervals(self) -> List[float]:
        """Return recent inter-keystroke intervals (ms) for identity/analytics."""
        return list(self._intervals)

    def reset(self) -> None:
        self._last_key_time = None
        self._intervals.clear()
        self._session_start = None
        self._last_keystroke_log.clear()
