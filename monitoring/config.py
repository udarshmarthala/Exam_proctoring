"""Tunable thresholds for the monitoring module (exam proctoring)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MonitoringConfig:
    # ----- Detection pipeline -----
    detection_interval_ms: float = 150.0  # 5–10 FPS
    smoothing_window_ms: float = 500.0
    confidence_threshold: float = 0.85
    grace_period_s: float = 60.0  # no alerting first 60s

    # ----- Gaze (iris ratio 0–1) -----
    gaze_left_threshold: float = 0.35
    gaze_right_threshold: float = 0.65
    gaze_up_threshold: float = 0.30
    gaze_down_threshold: float = 0.70
    gaze_off_center_deg: float = 15.0
    gaze_sustained_s: float = 1.5

    # ----- Head pose (degrees) -----
    head_yaw_warn_deg: float = 20.0   # L1 if sustained
    head_yaw_warn_sustained_s: float = 1.5
    head_yaw_alert_deg: float = 30.0  # L2 even brief
    head_pitch_down_warn_deg: float = 25.0
    head_pitch_down_sustained_s: float = 2.0
    head_pitch_up_warn_deg: float = 20.0
    head_pitch_up_sustained_s: float = 2.0
    # Legacy (for is_turned_away)
    head_yaw_threshold: float = 30.0
    head_pitch_threshold: float = 25.0

    # ----- Sustained violation durations (legacy / fallback) -----
    head_turned_duration: float = 1.5
    looking_away_duration: float = 1.5
    absence_duration: float = 3.0

    # ----- Eye closure / blink -----
    eye_closure_s: float = 2.0   # L1 if eyes closed > 2s
    blink_max_s: float = 0.4     # ignore if closed < 400ms
    eye_aspect_ratio_closed: float = 0.2  # below = closed

    # ----- Face absent / camera -----
    face_absent_l2_s: float = 3.0   # L2 after 3s no face
    face_absent_l3_s: float = 10.0  # L3 after 10s

    # ----- Alert UI -----
    level1_banner_dismiss_s: float = 4.0
    level2_cooldown_s: float = 30.0

    # ----- Escalation -----
    escalation_l1_count: int = 3
    escalation_l1_window_s: float = 5 * 60.0   # 5 min
    escalation_l1_expire_s: float = 10 * 60.0  # L1 older than 10 min don't count
    escalation_l2_count: int = 3  # 3 L2 total → L3

    # ----- Accessibility -----
    disable_yaw: bool = False
    disable_pitch: bool = False

    # ----- Sneezing heuristic: suppress eye_closure if head+eye resolve within 2s -----
    sneeze_resolve_s: float = 2.0

    # ----- MediaPipe -----
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    max_num_faces: int = 2

    # ----- Mouse tracking -----
    mouse_inactivity_s: float = 60.0       # L1 warning after no movement for 60s
    mouse_leave_alert_s: float = 2.0      # L2 if mouse outside window for 2s+
    mouse_erratic_speed_threshold: float = 2000.0   # px/s — above = erratic
    mouse_erratic_burst_count: int = 5    # movements in short window to flag erratic
    mouse_erratic_window_s: float = 0.5
    mouse_unusual_click_interval_ms: float = 80.0   # clicks < 80ms apart = double-click or bot-like
    mouse_unusual_click_count: int = 4    # repeated fast clicks to flag

    # ----- Keyboard tracking -----
    forbidden_shortcuts: tuple = ("Control+c", "Control+v", "Control+x", "Alt+Tab", "Meta", "Control+Tab")
    copy_paste_pause_ms: float = 500.0     # pause > 500ms after key suggests paste
    keyboard_inactivity_s: float = 120.0   # L1 after no keys for 2 min (exam-dependent)
    keystroke_dynamics_window: int = 20   # last N key intervals for dynamics (identity)
