"""
FastAPI route handlers for the Identity Verification Module.

Endpoints
---------
POST /enroll          — register a student's face
POST /verify          — run the LangGraph verification pipeline
POST /auth/validate   — validate a JWT token
GET  /health          — health check
GET  /students        — list enrolled students
"""
from __future__ import annotations

import logging
import shutil
import uuid
from datetime import timedelta
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.agents.graph import run_verification
from app.core.config import settings
from app.core.security import create_access_token, decode_access_token
from app.models.schemas import (
    AgentState,
    EnrollResponse,
    LivenessStatus,
    TokenValidateRequest,
    TokenValidateResponse,
    VerificationStatus,
    VerifyResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
MAX_FILE_SIZE_MB = 10


# ── Helpers ──────────────────────────────────────────────────────────────────

def _save_upload(upload: UploadFile, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)


def _validate_image_upload(file: UploadFile) -> None:
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported image type '{file.content_type}'. Use JPEG or PNG.",
        )


def _get_enrolled_path(student_id: str) -> Path | None:
    base = Path(settings.ENROLLED_FACES_DIR)
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = base / f"{student_id}{ext}"
        if p.exists():
            return p
    return None


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health", tags=["system"])
async def health_check():
    return {"status": "ok", "service": "exam-proctor-identity-verification"}


# ── Enrollment ────────────────────────────────────────────────────────────────

@router.post("/enroll", response_model=EnrollResponse, tags=["enrollment"])
async def enroll_student(
    student_id: str = Form(..., description="Unique student ID"),
    student_name: str = Form(..., description="Student full name"),
    photo: UploadFile = File(..., description="Clear frontal face photo"),
    id_document: UploadFile = File(None, description="Optional ID document photo"),
):
    """
    Enroll a new student by saving their reference face photo.
    Optionally stores an ID document for audit purposes.
    """
    _validate_image_upload(photo)

    # Determine extension
    ext_map = {
        "image/jpeg": ".jpg", "image/jpg": ".jpg",
        "image/png": ".png", "image/webp": ".webp",
    }
    ext = ext_map.get(photo.content_type, ".jpg")

    enrolled_dir = Path(settings.ENROLLED_FACES_DIR)
    face_path = enrolled_dir / f"{student_id}{ext}"

    # Check for duplicate
    existing = _get_enrolled_path(student_id)
    if existing:
        # Overwrite allowed — just log it
        logger.info("Re-enrolling student '%s', overwriting existing photo.", student_id)
        existing.unlink(missing_ok=True)

    _save_upload(photo, face_path)

    # Optionally save ID document
    if id_document:
        try:
            _validate_image_upload(id_document)
            id_ext = ext_map.get(id_document.content_type, ".jpg")
            id_path = enrolled_dir / f"{student_id}_id{id_ext}"
            _save_upload(id_document, id_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not save ID document for %s: %s", student_id, exc)

    # Quick face-detection check using DeepFace
    face_detected = False
    try:
        from deepface import DeepFace  # noqa: PLC0415
        faces = DeepFace.extract_faces(
            img_path=str(face_path),
            detector_backend=settings.DEEPFACE_DETECTOR,
            enforce_detection=True,
            align=True,
        )
        face_detected = len(faces) == 1
        if len(faces) == 0:
            face_path.unlink(missing_ok=True)
            return EnrollResponse(
                success=False,
                student_id=student_id,
                message="No face detected in the uploaded photo. Please upload a clear frontal face photo.",
                face_detected=False,
                embedding_stored=False,
            )
        if len(faces) > 1:
            face_path.unlink(missing_ok=True)
            return EnrollResponse(
                success=False,
                student_id=student_id,
                message=f"Multiple faces detected ({len(faces)}). Upload a photo with only one face.",
                face_detected=False,
                embedding_stored=False,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Face detection during enroll failed: %s", exc)
        face_detected = True  # Assume ok, verify at recognition time

    return EnrollResponse(
        success=True,
        student_id=student_id,
        message=f"Student '{student_name}' (ID: {student_id}) enrolled successfully.",
        face_detected=face_detected,
        embedding_stored=True,
    )


# ── Verification ──────────────────────────────────────────────────────────────

@router.post("/verify", response_model=VerifyResponse, tags=["verification"])
async def verify_student(
    student_id: str = Form(..., description="Student ID to verify against"),
    photo: UploadFile = File(..., description="Probe face photo (webcam capture or upload)"),
):
    """
    Run the full 4-node LangGraph verification pipeline for a student.
    Returns a JWT token on successful verification.
    """
    _validate_image_upload(photo)

    # Check student is enrolled
    enrolled_path = _get_enrolled_path(student_id)
    if not enrolled_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Student '{student_id}' is not enrolled. Please enroll first.",
        )

    # Save probe image temporarily
    ext_map = {
        "image/jpeg": ".jpg", "image/jpg": ".jpg",
        "image/png": ".png", "image/webp": ".webp",
    }
    ext = ext_map.get(photo.content_type, ".jpg")
    probe_path = Path(settings.UPLOAD_DIR) / f"probe_{uuid.uuid4().hex}{ext}"
    _save_upload(photo, probe_path)

    try:
        # Run LangGraph agent
        initial_state = {
            "student_id": student_id,
            "image_path": str(probe_path),
            "enrolled_image_path": str(enrolled_path),
        }
        result: AgentState = run_verification(initial_state)

        # Issue JWT on approval
        token: str | None = None
        if result.final_status == VerificationStatus.APPROVED:
            token = create_access_token(
                data={
                    "sub": student_id,
                    "status": result.final_status,
                    "confidence": result.final_confidence,
                },
                expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
            )

        return VerifyResponse(
            success=result.final_status == VerificationStatus.APPROVED,
            student_id=student_id,
            status=VerificationStatus(result.final_status),
            confidence=result.final_confidence,
            liveness=LivenessStatus(result.liveness_status),
            liveness_score=result.liveness_score,
            distance=result.recognition_distance,
            message=result.final_message,
            token=token,
            agent_trace=result.node_trace,
        )

    finally:
        # Clean up probe image
        probe_path.unlink(missing_ok=True)


# ── Auth Validate ─────────────────────────────────────────────────────────────

@router.post("/auth/validate", response_model=TokenValidateResponse, tags=["auth"])
async def validate_token(body: TokenValidateRequest):
    """
    Validate a JWT token issued by /verify.
    Returns token claims if valid, error detail if not.
    """
    payload = decode_access_token(body.token)
    if not payload:
        return TokenValidateResponse(
            valid=False,
            message="Token is invalid or has expired.",
        )

    from datetime import datetime, timezone  # noqa: PLC0415
    exp_ts = payload.get("exp")
    exp_str = (
        datetime.fromtimestamp(exp_ts, tz=timezone.utc).isoformat()
        if exp_ts else None
    )

    return TokenValidateResponse(
        valid=True,
        student_id=payload.get("sub"),
        student_name=payload.get("name"),
        expires_at=exp_str,
        message="Token is valid.",
    )


# ── Students list ─────────────────────────────────────────────────────────────

@router.get("/students", tags=["enrollment"])
async def list_students():
    """List all currently enrolled student IDs."""
    enrolled_dir = Path(settings.ENROLLED_FACES_DIR)
    students = []
    seen: set[str] = set()
    for p in sorted(enrolled_dir.glob("*")):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            sid = p.stem
            if "_id" not in sid and sid not in seen:
                students.append(sid)
                seen.add(sid)
    return {"enrolled_students": students, "count": len(students)}
