"""Data models for the monitoring module."""
from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class ProctoringEventType(str, Enum):
    gaze_deviation = "gaze_deviation"
    head_turn = "head_turn"
    face_absent = "face_absent"
    multiple_faces = "multiple_faces"
    eye_closure = "eye_closure"
    camera_blocked = "camera_blocked"
    # Mouse
    mouse_leave_window = "mouse_leave_window"
    mouse_erratic = "mouse_erratic"
    mouse_inactivity = "mouse_inactivity"
    mouse_unusual_clicks = "mouse_unusual_clicks"
    # Keyboard
    forbidden_shortcut = "forbidden_shortcut"
    copy_paste_suspected = "copy_paste_suspected"
    paste_used = "paste_used"
    keyboard_inactivity = "keyboard_inactivity"


class AlertLevel(int, Enum):
    L1 = 1  # soft warning
    L2 = 2  # hard warning
    L3 = 3  # critical


class GazeDirection(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    CENTER = "center"
    UP = "up"
    DOWN = "down"


class BehaviorFlag(str, Enum):
    LOOKING_AWAY = "looking_away"
    HEAD_TURNED = "head_turned"
    MULTIPLE_FACES = "multiple_faces"
    ABSENT = "absent"
    NORMAL = "normal"
    # Mouse/keyboard (for audit trail)
    MOUSE_LEAVE = "mouse_leave"
    MOUSE_ERRATIC = "mouse_erratic"
    MOUSE_INACTIVE = "mouse_inactive"
    MOUSE_UNUSUAL_CLICKS = "mouse_unusual_clicks"
    FORBIDDEN_SHORTCUT = "forbidden_shortcut"
    COPY_PASTE_SUSPECTED = "copy_paste_suspected"
    PASTE_USED = "paste_used"
    KEYBOARD_INACTIVE = "keyboard_inactive"


class EventSeverity(str, Enum):
    """Severity for audit trail: critical = immediate concern, warning = suspicious."""
    CRITICAL = "critical"
    WARNING = "warning"
    RESOLVED = "resolved"
    INFO = "info"


# Flags that warrant critical vs warning in the event log (proctoring best practice)
FLAG_SEVERITY: dict[BehaviorFlag, EventSeverity] = {
    BehaviorFlag.ABSENT: EventSeverity.CRITICAL,
    BehaviorFlag.MULTIPLE_FACES: EventSeverity.CRITICAL,
    BehaviorFlag.HEAD_TURNED: EventSeverity.WARNING,
    BehaviorFlag.LOOKING_AWAY: EventSeverity.WARNING,
    BehaviorFlag.NORMAL: EventSeverity.INFO,
    BehaviorFlag.MOUSE_LEAVE: EventSeverity.WARNING,
    BehaviorFlag.MOUSE_ERRATIC: EventSeverity.WARNING,
    BehaviorFlag.MOUSE_INACTIVE: EventSeverity.INFO,
    BehaviorFlag.MOUSE_UNUSUAL_CLICKS: EventSeverity.WARNING,
    BehaviorFlag.FORBIDDEN_SHORTCUT: EventSeverity.CRITICAL,
    BehaviorFlag.COPY_PASTE_SUSPECTED: EventSeverity.WARNING,
    BehaviorFlag.PASTE_USED: EventSeverity.CRITICAL,
    BehaviorFlag.KEYBOARD_INACTIVE: EventSeverity.INFO,
}

# Human-readable messages for audit log
EVENT_MESSAGES: dict[BehaviorFlag, str] = {
    BehaviorFlag.ABSENT: "No face detected (candidate may have left)",
    BehaviorFlag.MULTIPLE_FACES: "Multiple faces detected",
    BehaviorFlag.HEAD_TURNED: "Head turned away (sustained)",
    BehaviorFlag.LOOKING_AWAY: "Eyes not on screen (sustained)",
    BehaviorFlag.NORMAL: "Back to normal",
    BehaviorFlag.MOUSE_LEAVE: "Mouse left exam window",
    BehaviorFlag.MOUSE_ERRATIC: "Erratic or unusual mouse movement",
    BehaviorFlag.MOUSE_INACTIVE: "Prolonged mouse inactivity",
    BehaviorFlag.MOUSE_UNUSUAL_CLICKS: "Unusual click pattern detected",
    BehaviorFlag.FORBIDDEN_SHORTCUT: "Forbidden keyboard shortcut used",
    BehaviorFlag.COPY_PASTE_SUSPECTED: "Typing pause suggests copy-paste",
    BehaviorFlag.PASTE_USED: "Paste used (clipboard)",
    BehaviorFlag.KEYBOARD_INACTIVE: "Prolonged keyboard inactivity",
}


class MouseSnapshot(BaseModel):
    """Client-sent mouse state at a point in time."""
    timestamp: float = 0.0
    x: float = 0.0
    y: float = 0.0
    inside_window: bool = True
    buttons: int = 0  # bitmask: 1=left, 2=right, 4=middle


class MouseEventBatch(BaseModel):
    """Batch of mouse events (movements + clicks) since last frame."""
    movements: List[dict] = Field(default_factory=list)  # [{"t": float, "x": float, "y": float, "inside": bool}, ...]
    clicks: List[dict] = Field(default_factory=list)    # [{"t": float, "button": int, "x": float, "y": float}, ...]
    last_snapshot: Optional[dict] = None                # {timestamp, x, y, inside_window, buttons}


class KeystrokeRecord(BaseModel):
    """Single key event for dynamics and shortcut detection."""
    timestamp: float = 0.0
    key: str = ""
    code: str = ""
    keydown: bool = True
    ctrl: bool = False
    alt: bool = False
    meta: bool = False
    shift: bool = False


class MonitoringEvent(BaseModel):
    """Single entry in the session audit trail."""
    timestamp: float = 0.0
    flag: str = ""  # BehaviorFlag.value
    severity: str = ""  # EventSeverity.value
    message: str = ""


class ProctoringEvent(BaseModel):
    """Full proctoring event for log and export."""
    id: str = ""
    timestamp: str = ""  # ISO8601
    event_type: str = ""  # ProctoringEventType.value
    severity: int = 1  # 1 | 2 | 3
    duration_ms: float = 0.0
    head_yaw: Optional[float] = None
    head_pitch: Optional[float] = None
    gaze_vector: Optional[List[float]] = None  # [x, y]
    confidence_score: Optional[float] = None
    screenshot_ref: Optional[str] = None  # base64 or null
    dismissed: bool = False
    flagged_by_proctor: bool = False
    message: str = ""


class GazeResult(BaseModel):
    direction: GazeDirection = GazeDirection.CENTER
    horizontal_ratio: float = 0.5
    vertical_ratio: float = 0.5
    eye_aspect_ratio: float = 1.0  # for blink/closure; < 0.2 = closed


class HeadPoseResult(BaseModel):
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


class MonitoringFrame(BaseModel):
    timestamp: float = 0.0
    gaze: Optional[GazeResult] = None
    head_pose: Optional[HeadPoseResult] = None
    face_count: int = 0
    flags: List[BehaviorFlag] = Field(default_factory=list)
    # Proctoring: alerts fired this frame (level 1|2|3, event_type, metadata)
    alerts: List[dict] = Field(default_factory=list)
    # Smoothed values (after 500ms rolling average)
    confidence: float = 0.0
    low_light: bool = False
