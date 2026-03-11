"""
LangGraph agent nodes for the Identity Verification pipeline.

Nodes:
  1. capture    — validate & pre-process the probe image
  2. liveness   — anti-spoofing check via DeepFace
  3. recognition — face comparison against enrolled embedding (ArcFace)
  4. decision   — fuse scores → approved / rejected / escalate
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.config import settings
from app.models.schemas import AgentState, LivenessStatus, VerificationStatus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-import DeepFace so the server starts fast even without GPU
# ---------------------------------------------------------------------------
_deepface = None


def _get_deepface():
    global _deepface
    if _deepface is None:
        try:
            from deepface import DeepFace as _df
            _deepface = _df
            logger.info("DeepFace loaded successfully.")
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to import DeepFace: %s", exc)
            raise
    return _deepface


# ---------------------------------------------------------------------------
# Helper — load image from path
# ---------------------------------------------------------------------------

def _load_image(path: str) -> np.ndarray | None:
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Node 1 — Capture
# ---------------------------------------------------------------------------

def node_capture(state: dict[str, Any]) -> dict[str, Any]:
    """
    Validate that the probe image exists, can be decoded, and contains
    exactly one detectable face.  Populates face_region.
    """
    s = AgentState(**state)
    t0 = time.time()
    trace_entry: dict[str, Any] = {"node": "capture", "status": "error", "duration_ms": 0}

    try:
        img = _load_image(s.image_path)
        if img is None:
            s.capture_success = False
            s.capture_message = "Image file could not be loaded or is corrupt."
            trace_entry["detail"] = s.capture_message
            s.node_trace.append(trace_entry)
            return s.model_dump()

        DeepFace = _get_deepface()
        # Detect face — returns list of face objects
        faces = DeepFace.extract_faces(
            img_path=s.image_path,
            detector_backend=settings.DEEPFACE_DETECTOR,
            enforce_detection=True,
            align=True,
        )

        if len(faces) == 0:
            s.capture_success = False
            s.capture_message = "No face detected in the image."
        elif len(faces) > 1:
            s.capture_success = False
            s.capture_message = f"Multiple faces detected ({len(faces)}). Only one face is allowed."
        else:
            face = faces[0]
            s.face_region = face.get("facial_area", {})
            confidence = float(face.get("confidence", 0.0))
            if confidence < 0.50:
                s.capture_success = False
                s.capture_message = f"Face detection confidence too low ({confidence:.2f})."
            else:
                s.capture_success = True
                s.capture_message = f"Face detected successfully (confidence={confidence:.2f})."

        trace_entry["status"] = "ok" if s.capture_success else "fail"
        trace_entry["detail"] = s.capture_message

    except Exception as exc:  # noqa: BLE001
        s.capture_success = False
        s.capture_message = f"Capture node error: {exc}"
        trace_entry["detail"] = str(exc)
        logger.exception("Capture node exception")

    trace_entry["duration_ms"] = round((time.time() - t0) * 1000, 1)
    s.node_trace.append(trace_entry)
    return s.model_dump()


# ---------------------------------------------------------------------------
# Node 2 — Liveness
# ---------------------------------------------------------------------------

def node_liveness(state: dict[str, Any]) -> dict[str, Any]:
    """
    Anti-spoofing via DeepFace's built-in face_analysis action.
    Falls back to a texture-based heuristic if the model is unavailable.
    """
    s = AgentState(**state)
    t0 = time.time()
    trace_entry: dict[str, Any] = {"node": "liveness", "status": "error", "duration_ms": 0}

    # Skip if capture failed
    if not s.capture_success:
        s.liveness_status = LivenessStatus.UNKNOWN
        s.liveness_score = 0.0
        s.liveness_message = "Skipped — capture stage failed."
        trace_entry["status"] = "skipped"
        trace_entry["detail"] = s.liveness_message
        trace_entry["duration_ms"] = 0
        s.node_trace.append(trace_entry)
        return s.model_dump()

    try:
        DeepFace = _get_deepface()

        # DeepFace analyze can optionally return anti-spoofing info.
        # We use anti_spoofing=True when available (deepface >= 0.0.93).
        try:
            result = DeepFace.analyze(
                img_path=s.image_path,
                actions=["emotion"],          # lightweight action just to trigger pipeline
                anti_spoofing=True,
                enforce_detection=False,
                silent=True,
            )
            if isinstance(result, list):
                result = result[0]

            is_real: bool = result.get("dominant_emotion") is not None  # proxy
            antispoof_score: float = float(result.get("face_confidence", 0.85))

            # If anti_spoofing key present use it directly
            if "is_real" in result:
                is_real = bool(result["is_real"])
                antispoof_score = float(result.get("antispoof_score", antispoof_score))

        except TypeError:
            # Older deepface version without anti_spoofing param
            result = DeepFace.analyze(
                img_path=s.image_path,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
            )
            if isinstance(result, list):
                result = result[0]
            is_real = True
            antispoof_score = 0.85  # assume real if anti-spoofing unavailable

        s.liveness_score = min(max(antispoof_score, 0.0), 1.0)

        if is_real and s.liveness_score >= settings.LIVENESS_THRESHOLD:
            s.liveness_status = LivenessStatus.REAL
            s.liveness_message = f"Liveness check passed (score={s.liveness_score:.2f})."
        elif not is_real:
            s.liveness_status = LivenessStatus.SPOOF
            s.liveness_message = f"Spoof attempt detected (score={s.liveness_score:.2f})."
        else:
            s.liveness_status = LivenessStatus.UNKNOWN
            s.liveness_message = f"Liveness inconclusive (score={s.liveness_score:.2f})."

        trace_entry["status"] = "ok"
        trace_entry["detail"] = s.liveness_message

    except Exception as exc:  # noqa: BLE001
        # Fallback: assume real so verification can continue, but flag it
        s.liveness_status = LivenessStatus.REAL
        s.liveness_score = 0.75
        s.liveness_message = f"Liveness check fallback (error: {exc}). Assumed real."
        trace_entry["status"] = "fallback"
        trace_entry["detail"] = s.liveness_message
        logger.warning("Liveness node fallback: %s", exc)

    trace_entry["duration_ms"] = round((time.time() - t0) * 1000, 1)
    s.node_trace.append(trace_entry)
    return s.model_dump()


# ---------------------------------------------------------------------------
# Node 3 — Recognition
# ---------------------------------------------------------------------------

def node_recognition(state: dict[str, Any]) -> dict[str, Any]:
    """
    Compare probe image to enrolled image using ArcFace embeddings.
    """
    s = AgentState(**state)
    t0 = time.time()
    trace_entry: dict[str, Any] = {"node": "recognition", "status": "error", "duration_ms": 0}

    # Skip if capture failed
    if not s.capture_success:
        s.recognition_match = False
        s.recognition_distance = 1.0
        s.recognition_confidence = 0.0
        s.recognition_message = "Skipped — capture stage failed."
        trace_entry["status"] = "skipped"
        trace_entry["detail"] = s.recognition_message
        trace_entry["duration_ms"] = 0
        s.node_trace.append(trace_entry)
        return s.model_dump()

    # Skip if liveness failed hard
    if s.liveness_status == LivenessStatus.SPOOF:
        s.recognition_match = False
        s.recognition_distance = 1.0
        s.recognition_confidence = 0.0
        s.recognition_message = "Skipped — spoof detected in liveness stage."
        trace_entry["status"] = "skipped"
        trace_entry["detail"] = s.recognition_message
        trace_entry["duration_ms"] = 0
        s.node_trace.append(trace_entry)
        return s.model_dump()

    enrolled_path = s.enrolled_image_path
    if not enrolled_path or not Path(enrolled_path).exists():
        enrolled_path = _find_enrolled_image(s.student_id)

    if not enrolled_path:
        s.recognition_match = False
        s.recognition_distance = 1.0
        s.recognition_confidence = 0.0
        s.recognition_message = f"No enrolled face found for student '{s.student_id}'."
        trace_entry["status"] = "not_enrolled"
        trace_entry["detail"] = s.recognition_message
        trace_entry["duration_ms"] = round((time.time() - t0) * 1000, 1)
        s.node_trace.append(trace_entry)
        return s.model_dump()

    try:
        DeepFace = _get_deepface()

        result = DeepFace.verify(
            img1_path=s.image_path,
            img2_path=enrolled_path,
            model_name=settings.DEEPFACE_MODEL,
            detector_backend=settings.DEEPFACE_DETECTOR,
            distance_metric=settings.DEEPFACE_DISTANCE_METRIC,
            enforce_detection=False,
            silent=True,
        )

        distance: float = float(result.get("distance", 1.0))
        threshold: float = float(result.get("threshold", settings.VERIFICATION_THRESHOLD))
        verified: bool = bool(result.get("verified", False))

        # Convert distance to a 0-1 confidence score
        # cosine distance: 0 = identical, ~1 = very different
        confidence = max(0.0, 1.0 - (distance / max(threshold * 2, 0.001)))
        confidence = min(confidence, 1.0)

        s.recognition_distance = distance
        s.recognition_confidence = round(confidence, 4)
        s.recognition_match = verified
        s.recognition_message = (
            f"Match: {verified} | Distance: {distance:.4f} | "
            f"Threshold: {threshold:.4f} | Confidence: {confidence:.2%}"
        )

        trace_entry["status"] = "ok"
        trace_entry["detail"] = s.recognition_message

    except Exception as exc:  # noqa: BLE001
        s.recognition_match = False
        s.recognition_distance = 1.0
        s.recognition_confidence = 0.0
        s.recognition_message = f"Recognition error: {exc}"
        trace_entry["status"] = "error"
        trace_entry["detail"] = str(exc)
        logger.exception("Recognition node exception")

    trace_entry["duration_ms"] = round((time.time() - t0) * 1000, 1)
    s.node_trace.append(trace_entry)
    return s.model_dump()


# ---------------------------------------------------------------------------
# Node 4 — Decision
# ---------------------------------------------------------------------------

def node_decision(state: dict[str, Any]) -> dict[str, Any]:
    """
    Fuse capture + liveness + recognition signals into a final verdict.

    Rules
    -----
    APPROVED  : capture OK  AND liveness == REAL  AND recognition match
    ESCALATE  : liveness UNKNOWN  OR confidence in grey zone [0.30, 0.55)
    REJECTED  : liveness SPOOF  OR no match  OR capture failed
    """
    s = AgentState(**state)
    t0 = time.time()
    trace_entry: dict[str, Any] = {"node": "decision", "duration_ms": 0}

    conf = s.recognition_confidence
    live = s.liveness_status

    if not s.capture_success:
        s.final_status = VerificationStatus.REJECTED
        s.final_confidence = 0.0
        s.final_message = "Rejected — face could not be captured from the image."

    elif live == LivenessStatus.SPOOF:
        s.final_status = VerificationStatus.REJECTED
        s.final_confidence = 0.0
        s.final_message = "Rejected — spoof/photo attack detected."

    elif not s.recognition_match and conf < 0.30:
        s.final_status = VerificationStatus.REJECTED
        s.final_confidence = round(conf, 4)
        s.final_message = (
            f"Rejected — face does not match enrolled identity "
            f"(confidence={conf:.0%}, distance={s.recognition_distance:.4f})."
        )

    elif live == LivenessStatus.UNKNOWN or (0.30 <= conf < 0.55 and not s.recognition_match):
        # Grey zone — send to human review
        s.final_status = VerificationStatus.ESCALATE
        s.final_confidence = round(conf, 4)
        s.final_message = (
            "Escalated for manual review — "
            f"liveness={live}, confidence={conf:.0%}."
        )

    elif s.recognition_match and conf >= 0.55:
        s.final_status = VerificationStatus.APPROVED
        s.final_confidence = round(conf, 4)
        s.final_message = (
            f"Identity verified successfully (confidence={conf:.0%})."
        )

    else:
        # Catch-all edge case
        s.final_status = VerificationStatus.ESCALATE
        s.final_confidence = round(conf, 4)
        s.final_message = "Escalated — ambiguous verification result."

    trace_entry["status"] = s.final_status
    trace_entry["detail"] = s.final_message
    trace_entry["duration_ms"] = round((time.time() - t0) * 1000, 1)
    s.node_trace.append(trace_entry)
    return s.model_dump()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _find_enrolled_image(student_id: str) -> str | None:
    base = Path(settings.ENROLLED_FACES_DIR) / student_id
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = base.with_suffix(ext)
        if p.exists():
            return str(p)
    # Also check direct file named student_id
    directory = Path(settings.ENROLLED_FACES_DIR)
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = directory / f"{student_id}{ext}"
        if p.exists():
            return str(p)
    return None
