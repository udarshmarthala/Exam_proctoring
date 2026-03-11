"""
Pydantic schemas for request/response models.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────────

class VerificationStatus(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATE = "escalate"


class LivenessStatus(str, Enum):
    REAL = "real"
    SPOOF = "spoof"
    UNKNOWN = "unknown"


# ── Enrollment ───────────────────────────────────────────────────────────────

class EnrollRequest(BaseModel):
    student_id: str = Field(..., min_length=3, max_length=64, description="Unique student identifier")
    student_name: str = Field(..., min_length=2, max_length=128)


class EnrollResponse(BaseModel):
    success: bool
    student_id: str
    message: str
    face_detected: bool = False
    embedding_stored: bool = False


# ── Verification ─────────────────────────────────────────────────────────────

class VerifyResponse(BaseModel):
    success: bool
    student_id: str
    status: VerificationStatus
    confidence: float = Field(..., ge=0.0, le=1.0)
    liveness: LivenessStatus
    liveness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    distance: float = Field(default=1.0)
    message: str
    token: str | None = None
    agent_trace: list[dict[str, Any]] = Field(default_factory=list)


# ── Auth ─────────────────────────────────────────────────────────────────────

class TokenValidateRequest(BaseModel):
    token: str


class TokenValidateResponse(BaseModel):
    valid: bool
    student_id: str | None = None
    student_name: str | None = None
    expires_at: str | None = None
    message: str


# ── Agent State ──────────────────────────────────────────────────────────────

class AgentState(BaseModel):
    """Shared state object passed between LangGraph nodes."""
    student_id: str = ""
    image_path: str = ""
    enrolled_image_path: str = ""

    # Capture node output
    capture_success: bool = False
    capture_message: str = ""
    face_region: dict[str, Any] | None = None

    # Liveness node output
    liveness_status: LivenessStatus = LivenessStatus.UNKNOWN
    liveness_score: float = 0.0
    liveness_message: str = ""

    # Recognition node output
    recognition_match: bool = False
    recognition_distance: float = 1.0
    recognition_confidence: float = 0.0
    recognition_message: str = ""

    # Decision node output
    final_status: VerificationStatus = VerificationStatus.REJECTED
    final_message: str = ""
    final_confidence: float = 0.0

    # Trace
    node_trace: list[dict[str, Any]] = Field(default_factory=list)

    class Config:
        use_enum_values = True
