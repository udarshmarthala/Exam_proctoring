"""
Monitoring API routes — extracted from monitoring/server.py for integration
into the main FastAPI app.

All routes are mounted under prefix="/monitoring".
"""
from __future__ import annotations

import base64
import json
import os
import threading
import uuid as uuid_lib
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from concurrent.futures import ThreadPoolExecutor

# Optional ultralytics YOLO for phone detection
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

from monitoring.models import (
    BehaviorFlag,
    EventSeverity,
    EVENT_MESSAGES,
    FLAG_SEVERITY,
    ProctoringEventType,
)

# Map proctoring event_type to BehaviorFlag for event_log
EVENT_TYPE_TO_FLAG = {
    "mouse_leave_window": BehaviorFlag.MOUSE_LEAVE,
    "mouse_erratic": BehaviorFlag.MOUSE_ERRATIC,
    "mouse_inactivity": BehaviorFlag.MOUSE_INACTIVE,
    "mouse_unusual_clicks": BehaviorFlag.MOUSE_UNUSUAL_CLICKS,
    "forbidden_shortcut": BehaviorFlag.FORBIDDEN_SHORTCUT,
    "copy_paste_suspected": BehaviorFlag.COPY_PASTE_SUSPECTED,
    "paste_used": BehaviorFlag.PASTE_USED,
    "keyboard_inactivity": BehaviorFlag.KEYBOARD_INACTIVE,
    "talking_detected": BehaviorFlag.TALKING,
    "document_face_mismatch": BehaviorFlag.DOCUMENT_MISMATCH,
}

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

_monitor = None
_monitor_error: str | None = None
_monitor_lock = threading.Lock()
_monitor_warmup_started = False

# Module-level YOLO model and executor (lazy-initialized)
_PHONE_YOLO = None
_PHONE_EXECUTOR = None


def get_monitor():
    """Lazy-init: load MediaPipe/monitor only on first use."""
    global _monitor, _monitor_error
    if _monitor is not None:
        return _monitor, None
    if _monitor_error is not None:
        return None, _monitor_error
    with _monitor_lock:
        if _monitor is not None:
            return _monitor, None
        if _monitor_error is not None:
            return None, _monitor_error
        try:
            from monitoring.behavior_monitor import BehaviorMonitor
            from monitoring.config import MonitoringConfig
            _monitor = BehaviorMonitor(MonitoringConfig())
            return _monitor, None
        except Exception as e:
            _monitor_error = str(e)
            return None, _monitor_error


def _background_warmup_monitor() -> None:
    """Best-effort background warmup so first WS frame is not blocked by model load."""
    get_monitor()


def _start_monitor_warmup_if_needed() -> str:
    global _monitor_warmup_started
    if _monitor is not None:
        return "ready"
    if _monitor_error is not None:
        return "error"

    with _monitor_lock:
        if _monitor is not None:
            return "ready"
        if _monitor_error is not None:
            return "error"
        if not _monitor_warmup_started:
            _monitor_warmup_started = True
            t = threading.Thread(target=_background_warmup_monitor, daemon=True)
            t.start()
        return "warming"


@router.get("/warmup")
async def warmup_monitor():
    """Trigger/inspect monitor model warmup state for faster exam-start UX."""
    status = _start_monitor_warmup_if_needed()
    return {
        "status": status,
        "ready": _monitor is not None,
        "error": _monitor_error,
    }


_mouse_tracker = None
_keyboard_tracker = None


def get_mouse_tracker():
    global _mouse_tracker
    if _mouse_tracker is None:
        from monitoring.mouse_tracker import MouseTracker
        from monitoring.config import MonitoringConfig
        _mouse_tracker = MouseTracker(MonitoringConfig())
    return _mouse_tracker


def get_keyboard_tracker():
    global _keyboard_tracker
    if _keyboard_tracker is None:
        from monitoring.keyboard_tracker import KeyboardTracker
        from monitoring.config import MonitoringConfig
        _keyboard_tracker = KeyboardTracker(MonitoringConfig())
    return _keyboard_tracker


# ── Document face verification ──────────────────────────────────────────────
_document_faces: dict[str, str] = {}
DOCUMENT_DIR = Path("data/documents")
DOCUMENT_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload-document")
async def upload_document(
    student_id: str = Form(...),
    document: UploadFile = File(...),
):
    """Upload an ID document, extract the face from it, and save for periodic matching."""
    import tempfile
    contents = await document.read()
    if not contents:
        return JSONResponse(status_code=400, content={"success": False, "message": "Empty file"})

    arr = np.frombuffer(contents, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(status_code=400, content={"success": False, "message": "Cannot decode image"})

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            cv2.imwrite(tmp.name, img)
            tmp_path = tmp.name
        from deepface import DeepFace
        faces = DeepFace.extract_faces(
            img_path=tmp_path,
            detector_backend="opencv",
            enforce_detection=True,
            align=True,
        )
        if not faces:
            return JSONResponse(status_code=400, content={"success": False, "message": "No face found in document"})
        face = faces[0]
        confidence = face.get("confidence", 0)
        if confidence < 0.50:
            return JSONResponse(status_code=400, content={
                "success": False,
                "message": f"Face confidence too low ({confidence:.2f}). Please upload a clearer document.",
            })
    except Exception as e:
        return JSONResponse(status_code=400, content={"success": False, "message": f"Face extraction failed: {e}"})
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    doc_path = DOCUMENT_DIR / f"{student_id}_doc.jpg"
    cv2.imwrite(str(doc_path), img)
    _document_faces[student_id] = str(doc_path)
    return {"success": True, "message": "Document uploaded — face extracted for verification", "student_id": student_id}


@router.post("/upload-document-base64")
async def upload_document_base64(request: dict):
    """Upload an ID document as base64 (used by the webcam capture UI)."""
    student_id = request.get("student_id", "").strip()
    image_b64 = request.get("image", "").strip()
    if not student_id or not image_b64:
        return JSONResponse(status_code=400, content={"success": False, "message": "student_id and image required"})
    try:
        img_data = base64.b64decode(image_b64)
    except Exception:
        return JSONResponse(status_code=400, content={"success": False, "message": "Invalid base64"})

    arr = np.frombuffer(img_data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(status_code=400, content={"success": False, "message": "Cannot decode image"})

    import tempfile
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            cv2.imwrite(tmp.name, img)
            tmp_path = tmp.name
        from deepface import DeepFace
        faces = DeepFace.extract_faces(
            img_path=tmp_path,
            detector_backend="opencv",
            enforce_detection=True,
            align=True,
        )
        if not faces:
            return JSONResponse(status_code=400, content={"success": False, "message": "No face found in document"})
        face = faces[0]
        confidence = face.get("confidence", 0)
        if confidence < 0.50:
            return JSONResponse(status_code=400, content={
                "success": False,
                "message": f"Face confidence too low ({confidence:.2f}). Upload a clearer document.",
            })
    except Exception as e:
        return JSONResponse(status_code=400, content={"success": False, "message": f"Face extraction failed: {e}"})
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    doc_path = DOCUMENT_DIR / f"{student_id}_doc.jpg"
    cv2.imwrite(str(doc_path), img)
    _document_faces[student_id] = str(doc_path)
    return {"success": True, "message": "Document uploaded — face extracted for verification", "student_id": student_id}


def _find_document_face(student_id: str) -> str | None:
    """Return path to document face image if it exists."""
    if student_id in _document_faces:
        p = _document_faces[student_id]
        if Path(p).exists():
            return p
    for ext in ('.jpg', '.jpeg', '.png'):
        p = DOCUMENT_DIR / f"{student_id}_doc{ext}"
        if p.exists():
            _document_faces[student_id] = str(p)
            return str(p)
    return None


@router.get("/document-preview")
async def document_preview(student_id: str = ""):
    """Serve the uploaded ID document image for the given student_id."""
    student_id = (student_id or "").strip()
    if not student_id:
        return Response(status_code=400)
    path = _find_document_face(student_id)
    if not path or not Path(path).exists():
        return Response(status_code=404)
    return FileResponse(path, media_type="image/jpeg")


_COCO_PHONE_CLASS_ID = 67  # COCO class 67 = "cell phone"
_PHONE_CLASS_KEYWORDS = ('phone', 'cell', 'mobile', 'smartphone')


def _phone_detection_worker(frame_bgr, conf_thresh=0.35):
    """Run YOLO model on frame; return (detected: bool, confidence: float)."""
    global _PHONE_YOLO
    if YOLO is None:
        return False, 0.0
    try:
        if _PHONE_YOLO is None:
            _PHONE_YOLO = YOLO('yolov8n.pt')
    except Exception:
        return False, 0.0
    try:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = _PHONE_YOLO(rgb, imgsz=416, conf=conf_thresh, verbose=False)
        if not results:
            return False, 0.0
        r = results[0]
        boxes = getattr(r, 'boxes', None)
        if boxes is None:
            return False, 0.0
        cls_ids = getattr(boxes, 'cls', None)
        confs = getattr(boxes, 'conf', None)
        if cls_ids is None:
            return False, 0.0
        best_conf = 0.0
        for i, cid in enumerate(cls_ids):
            try:
                idx = int(cid.item()) if hasattr(cid, 'item') else int(cid)
            except Exception:
                continue
            conf = 0.0
            try:
                conf = float(confs[i].item()) if hasattr(confs[i], 'item') else float(confs[i])
            except Exception:
                conf = 0.0
            # Match by COCO class ID 67 OR by class name keywords
            name = ''
            try:
                name = (_PHONE_YOLO.names.get(idx, '') if hasattr(_PHONE_YOLO, 'names') else '').lower()
            except Exception:
                name = ''
            is_phone = (idx == _COCO_PHONE_CLASS_ID) or any(kw in name for kw in _PHONE_CLASS_KEYWORDS)
            if is_phone and conf >= conf_thresh:
                best_conf = max(best_conf, conf)
        if best_conf > 0.0:
            return True, best_conf
        return False, 0.0
    except Exception:
        return False, 0.0


# ── Monitoring HTML page ─────────────────────────────────────────────────────

_MONITORING_HTML_PATH = Path(__file__).parent.parent / "static" / "monitoring.html"


@router.get("/exam", response_class=HTMLResponse)
async def monitoring_page():
    """Serve the monitoring/exam page."""
    return HTMLResponse(_MONITORING_HTML_PATH.read_text(encoding="utf-8"))


# ── Helpers ──────────────────────────────────────────────────────────────────

def _event_entry(timestamp: float, flag: BehaviorFlag, severity: EventSeverity, message: str) -> dict:
    return {"timestamp": timestamp, "flag": flag.value, "severity": severity.value, "message": message}


def _capture_screenshot(frame) -> str | None:
    if frame is None or frame.size == 0:
        return None
    _, buf = cv2.imencode(".jpg", frame)
    return base64.b64encode(buf.tobytes()).decode("ascii")


_FACE_CASCADE = None

def _extract_face_crop(frame, padding: float = 0.20):
    """
    Detect the largest face in frame and return a tightly-cropped BGR image.
    padding adds extra margin around the detected region so DeepFace has
    enough context.  Returns None if no face is found.
    """
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _FACE_CASCADE = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )
    if len(faces) == 0:
        return None
    # Pick the largest detected face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    fh, fw = frame.shape[:2]
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(fw, x + w + pad_x)
    y2 = min(fh, y + h + pad_y)
    return frame[y1:y2, x1:x2].copy()


def _parse_incoming(raw: str):
    """Parse incoming WS message: JSON with frame + optional mouse/keys/paste_event/audio_level + student_id + student_name."""
    raw = raw.strip()
    if raw.startswith("{"):
        try:
            obj = json.loads(raw)
            frame_b64 = obj.get("frame")
            mouse = obj.get("mouse")
            keys = obj.get("keys")
            paste_event = bool(obj.get("paste_event"))
            audio_level = float(obj.get("audio_level", 0.0))
            student_id = obj.get("student_id") if "student_id" in obj else None
            student_name = (obj.get("student_name") or "").strip() or None
            return frame_b64, mouse, keys, paste_event, audio_level, student_id, student_name
        except json.JSONDecodeError:
            pass
    return raw, None, None, False, 0.0, None, None


def _find_enrolled_image(student_id: str) -> str | None:
    """Return path to enrolled image for student_id if it exists."""
    try:
        from app.core.config import settings
        base = Path(settings.ENROLLED_FACES_DIR)
    except Exception:
        return None
    for ext in ('.jpg', '.jpeg', '.png', '.webp'):
        p = base / f"{student_id}{ext}"
        if p.exists():
            return str(p)
    return None


def _get_enrolled_students() -> list[dict]:
    """Return list of enrolled students {id, name} from main app registry."""
    try:
        from app.core.config import settings
        base = Path(settings.ENROLLED_FACES_DIR)
        registry_path = base / "registry.json"
        registry: dict = {}
        if registry_path.exists():
            with open(registry_path, encoding="utf-8") as f:
                registry = json.load(f)
        seen: set[str] = set()
        out: list[dict] = []
        for p in sorted(base.glob("*")):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} and "_id" not in p.stem:
                sid = p.stem
                if sid not in seen:
                    seen.add(sid)
                    out.append({"id": sid, "name": registry.get(sid) or sid})
        return out
    except Exception:
        return []


# ── WebSocket endpoint ───────────────────────────────────────────────────────

@router.websocket("/ws/monitor")
async def ws_monitor(websocket: WebSocket):
    await websocket.accept()
    event_log: list[dict] = []
    prev_flags: set[str] = set()
    max_log_entries = 500
    proctoring_events: list[dict] = []
    session_start: float | None = None
    last_l2_time: float = 0.0
    L2_COOLDOWN_S = 30.0
    # Multiple-faces detection (independent 1-second cooldown)
    last_multiple_faces_alert: float = 0.0
    MULTIPLE_FACES_COOLDOWN_S: float = 1.0
    # Server-side re-verification state
    reverify_last_time: float = 0.0
    reverify_failures: int = 0
    REVERIFY_INTERVAL_S: float = 12.0  # verify identity every 12 seconds
    session_ref_photo_path: Path | None = None  # live photo captured at exam start
    # Phone detection state
    phone_future = None
    phone_detect_start: float | None = None
    last_phone_alert: float = 0.0
    PHONE_CHECK_INTERVAL_S = 0.5
    PHONE_DEBOUNCE_S = 1.5        # seconds phone must be visible before first alert
    PHONE_ALERT_COOLDOWN_S = 10.0  # re-alert every 10s while phone stays in frame
    PHONE_CONF_THRESH = 0.30       # lowered from 0.35 for better recall
    last_phone_check_time: float = 0.0
    # Voice activity detection state
    VOICE_THRESHOLD = 0.015
    voice_detected = False
    voice_start: float = 0.0
    VOICE_SUSTAINED_S = 1.0
    VOICE_COOLDOWN_S = 8.0
    last_voice_alert: float = 0.0
    # Document face verification state (every 1 minute)
    DOC_VERIFY_INTERVAL_S: float = 60.0
    doc_verify_last_time: float = 0.0
    doc_verify_failures: int = 0
    session_student_id: str | None = None
    session_student_name: str | None = None

    try:
        while True:
            raw = await websocket.receive_text()
            frame_b64, mouse_data, key_events, paste_event, audio_level, student_id, student_name = _parse_incoming(raw)
            if student_id and session_student_id is None:
                session_student_id = (student_id or "").strip() or None
            if student_name:
                session_student_name = student_name
            elif session_student_id and session_student_name is None:
                for s in _get_enrolled_students():
                    if s.get("id") == session_student_id:
                        session_student_name = s.get("name") or session_student_id
                        break
                if session_student_name is None:
                    session_student_name = session_student_id
            if not frame_b64:
                await websocket.send_json({"error": "Missing frame data"})
                continue
            img_bytes = base64.b64decode(frame_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_json({"error": "Could not decode frame"})
                continue

            frame = cv2.flip(frame, 1)
            import time as time_mod
            now = time_mod.time()
            if session_start is None:
                session_start = now

            # Capture the reference photo from the very first frame of the exam (face crop only)
            if session_ref_photo_path is None and student_id:
                try:
                    face_crop = _extract_face_crop(frame)
                    if face_crop is not None:
                        from app.core.config import settings
                        ref_path = Path(settings.UPLOAD_DIR) / f"ref_{student_id}_{uuid_lib.uuid4().hex}.jpg"
                        cv2.imwrite(str(ref_path), face_crop)
                        session_ref_photo_path = ref_path
                except Exception:
                    pass

            monitor, err = get_monitor()
            if err is not None:
                await websocket.send_json({
                    "error": "Monitoring failed to start",
                    "detail": err,
                    "hint": "Try running the server in a normal Terminal (not Cursor) or use a Python from Homebrew: brew install python@3.11 && python3.11 -m venv .venv && source .venv/bin/activate",
                })
                continue
            result = monitor.process_frame(frame)
            current_flags = {f.value for f in result.flags if f != BehaviorFlag.NORMAL}
            new_events: list[dict] = []

            # Mouse and keyboard tracking
            extra_alerts: list[dict] = []
            if mouse_data:
                mt = get_mouse_tracker()
                movements = mouse_data.get("movements") or []
                clicks = mouse_data.get("clicks") or []
                last_snapshot = mouse_data.get("last_snapshot")
                extra_alerts.extend(mt.process_batch(movements, clicks, last_snapshot, now))
            if key_events:
                kt = get_keyboard_tracker()
                kt_alerts, _ = kt.process_batch(key_events, now)
                extra_alerts.extend(kt_alerts)
            if paste_event:
                extra_alerts.append({
                    "level": 2,
                    "event_type": ProctoringEventType.paste_used.value,
                    "message": "Paste used (clipboard)",
                    "duration_ms": 0,
                    "confidence_score": 1.0,
                })

            for f in result.flags:
                if f == BehaviorFlag.NORMAL:
                    continue
                if f.value not in prev_flags:
                    severity = FLAG_SEVERITY.get(f, EventSeverity.WARNING)
                    msg = EVENT_MESSAGES.get(f, f.value.replace("_", " ").title())
                    entry = _event_entry(result.timestamp, f, severity, msg)
                    event_log.append(entry)
                    new_events.append(entry)
                    if len(event_log) > max_log_entries:
                        event_log.pop(0)

            for prev in prev_flags:
                if prev not in current_flags:
                    entry = _event_entry(
                        result.timestamp,
                        BehaviorFlag.NORMAL,
                        EventSeverity.RESOLVED,
                        EVENT_MESSAGES[BehaviorFlag.NORMAL],
                    )
                    event_log.append(entry)
                    new_events.append(entry)
                    if len(event_log) > max_log_entries:
                        event_log.pop(0)
                    break
            prev_flags = current_flags

            # Proctoring alerts
            alerts_to_send: list[dict] = []
            # Multiple faces check — runs every ~1 second with its own cooldown
            try:
                face_count = getattr(result, 'face_count', 0)
                if face_count > 1 and (now - last_multiple_faces_alert) >= MULTIPLE_FACES_COOLDOWN_S:
                    last_multiple_faces_alert = now
                    screenshot_ref = _capture_screenshot(frame)
                    pe = {
                        "id": str(uuid_lib.uuid4()),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "event_type": "multiple_faces",
                        "severity": 2,
                        "duration_ms": 0,
                        "head_yaw": None,
                        "head_pitch": None,
                        "gaze_vector": None,
                        "confidence_score": 0.0,
                        "screenshot_ref": screenshot_ref,
                        "dismissed": False,
                        "flagged_by_proctor": False,
                        "message": f"Multiple faces detected ({face_count})",
                    }
                    proctoring_events.append(pe)
                    if len(proctoring_events) > 500:
                        proctoring_events.pop(0)
                    alerts_to_send.append({"level": 2, "event_type": "multiple_faces", "message": pe["message"], "id": pe["id"]})
                    entry = _event_entry(now, BehaviorFlag.MULTIPLE_FACES, EventSeverity.WARNING, pe["message"])
                    event_log.append(entry)
                    new_events.append(entry)
                    if len(event_log) > max_log_entries:
                        event_log.pop(0)
            except Exception:
                pass

            all_alerts = list(getattr(result, "alerts", []) or []) + extra_alerts
            for al in all_alerts:
                level = al.get("level", 1)
                if level == 2 and (now - last_l2_time) < L2_COOLDOWN_S:
                    continue
                if level == 2:
                    last_l2_time = now
                screenshot_ref = _capture_screenshot(frame) if level >= 2 else None
                pe = {
                    "id": str(uuid_lib.uuid4()),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event_type": al.get("event_type", "gaze_deviation"),
                    "severity": level,
                    "duration_ms": al.get("duration_ms", 0),
                    "head_yaw": al.get("head_yaw"),
                    "head_pitch": al.get("head_pitch"),
                    "gaze_vector": al.get("gaze_vector"),
                    "confidence_score": al.get("confidence_score"),
                    "screenshot_ref": screenshot_ref,
                    "dismissed": False,
                    "flagged_by_proctor": False,
                    "message": al.get("message", ""),
                }
                proctoring_events.append(pe)
                if len(proctoring_events) > 500:
                    proctoring_events.pop(0)
                alerts_to_send.append({**al, "id": pe["id"]})

            # Audit trail for mouse/keyboard alerts
            for al in extra_alerts:
                flag = EVENT_TYPE_TO_FLAG.get(al.get("event_type"))
                if flag is not None:
                    severity = FLAG_SEVERITY.get(flag, EventSeverity.WARNING)
                    msg = al.get("message", EVENT_MESSAGES.get(flag, al.get("event_type", "")))
                    entry = _event_entry(result.timestamp, flag, severity, msg)
                    event_log.append(entry)
                    new_events.append(entry)
                    if len(event_log) > max_log_entries:
                        event_log.pop(0)

            # Server-side periodic re-verification (every 12s against session start photo)
            if student_id and session_ref_photo_path is not None:
                try:
                    if (now - reverify_last_time) >= REVERIFY_INTERVAL_S:
                        reverify_last_time = now
                        try:
                            from app.core.config import settings
                            probe_crop = _extract_face_crop(frame)
                            if probe_crop is None:
                                probe_crop = frame  # fall back to full frame if detector misses
                            probe_path = Path(settings.UPLOAD_DIR) / f"probe_ws_{uuid_lib.uuid4().hex}.jpg"
                            cv2.imwrite(str(probe_path), probe_crop)
                            from deepface import DeepFace
                            v = DeepFace.verify(
                                img1_path=str(probe_path),
                                img2_path=str(session_ref_photo_path),
                                model_name=settings.DEEPFACE_MODEL,
                                detector_backend=settings.DEEPFACE_DETECTOR,
                                distance_metric=settings.DEEPFACE_DISTANCE_METRIC,
                                enforce_detection=False,
                                silent=True,
                            )
                            verified = bool(v.get("verified", False))
                            distance = float(v.get("distance", 1.0))
                            threshold = float(v.get("threshold", settings.VERIFICATION_THRESHOLD))
                            if not verified:
                                reverify_failures += 1
                                screenshot_ref = _capture_screenshot(frame)
                                pe = {
                                    "id": str(uuid_lib.uuid4()),
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "event_type": "identity_mismatch",
                                    "severity": 2,
                                    "duration_ms": 0,
                                    "head_yaw": None,
                                    "head_pitch": None,
                                    "gaze_vector": None,
                                    "confidence_score": 1.0 - min(max(distance / max(threshold * 2, 0.001), 0.0), 1.0),
                                    "screenshot_ref": screenshot_ref,
                                    "dismissed": False,
                                    "flagged_by_proctor": False,
                                    "message": f"Identity mismatch detected (distance={distance:.4f})",
                                }
                                proctoring_events.append(pe)
                                if len(proctoring_events) > 500:
                                    proctoring_events.pop(0)
                                alerts_to_send.append({"level": 2, "event_type": "identity_mismatch", "message": pe["message"], "id": pe["id"]})
                            else:
                                if reverify_failures > 0:
                                    entry = _event_entry(now, BehaviorFlag.NORMAL, EventSeverity.RESOLVED, "Identity re-verified")
                                    event_log.append(entry)
                                    new_events.append(entry)
                                reverify_failures = 0
                        finally:
                            try:
                                probe_path.unlink(missing_ok=True)
                            except Exception:
                                pass
                except Exception as exc:
                    try:
                        entry = _event_entry(now, BehaviorFlag.NORMAL, EventSeverity.WARNING, f"Reverify error: {exc}")
                        event_log.append(entry)
                        new_events.append(entry)
                    except Exception:
                        pass

            # Phone detection (background YOLO)
            try:
                global _PHONE_EXECUTOR
                if _PHONE_EXECUTOR is None:
                    _PHONE_EXECUTOR = ThreadPoolExecutor(max_workers=1)

                if phone_future is not None and phone_future.done():
                    detected = False
                    phone_conf = 0.0
                    try:
                        result_val = phone_future.result(timeout=0)
                        if isinstance(result_val, tuple):
                            detected, phone_conf = result_val
                        else:
                            detected = bool(result_val)
                    except Exception:
                        detected = False
                    phone_future = None
                    if detected:
                        if phone_detect_start is None:
                            phone_detect_start = now
                        debounced = (now - phone_detect_start) >= PHONE_DEBOUNCE_S
                        cooldown_ok = (now - last_phone_alert) >= PHONE_ALERT_COOLDOWN_S
                        if debounced and cooldown_ok:
                            last_phone_alert = now
                            screenshot_ref = _capture_screenshot(frame)
                            pe = {
                                "id": str(uuid_lib.uuid4()),
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "event_type": ProctoringEventType.possible_phone.value,
                                "severity": 2,
                                "duration_ms": 0,
                                "head_yaw": None,
                                "head_pitch": None,
                                "gaze_vector": None,
                                "confidence_score": round(phone_conf, 3),
                                "screenshot_ref": screenshot_ref,
                                "dismissed": False,
                                "flagged_by_proctor": False,
                                "message": f"Phone detected (conf={phone_conf:.2f})",
                            }
                            proctoring_events.append(pe)
                            if len(proctoring_events) > 500:
                                proctoring_events.pop(0)
                            alerts_to_send.append({"level": 2, "event_type": ProctoringEventType.possible_phone.value, "message": pe["message"], "id": pe["id"]})
                            entry = _event_entry(now, BehaviorFlag.NORMAL, EventSeverity.WARNING, pe["message"])
                            event_log.append(entry)
                            new_events.append(entry)
                    else:
                        phone_detect_start = None

                if phone_future is None and (now - last_phone_check_time) >= PHONE_CHECK_INTERVAL_S and YOLO is not None:
                    last_phone_check_time = now
                    try:
                        phone_future = _PHONE_EXECUTOR.submit(_phone_detection_worker, frame.copy(), PHONE_CONF_THRESH)
                    except Exception:
                        phone_future = None
            except Exception:
                pass

            # Voice Activity Detection
            voice_detected = audio_level >= VOICE_THRESHOLD
            if voice_detected:
                if voice_start == 0.0:
                    voice_start = now
                elif (now - voice_start) >= VOICE_SUSTAINED_S and (now - last_voice_alert) >= VOICE_COOLDOWN_S:
                    last_voice_alert = now
                    voice_duration = now - voice_start
                    voice_start = 0.0
                    screenshot_ref = _capture_screenshot(frame)
                    pe = {
                        "id": str(uuid_lib.uuid4()),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "event_type": "talking_detected",
                        "severity": 2,
                        "duration_ms": voice_duration * 1000,
                        "head_yaw": None,
                        "head_pitch": None,
                        "gaze_vector": None,
                        "confidence_score": min(audio_level * 10, 1.0),
                        "screenshot_ref": screenshot_ref,
                        "dismissed": False,
                        "flagged_by_proctor": False,
                        "message": f"Voice detected — audio level {audio_level:.3f}",
                    }
                    proctoring_events.append(pe)
                    alerts_to_send.append({
                        "level": 2,
                        "event_type": "talking_detected",
                        "message": "Voice detected — please remain silent during the exam",
                        "duration_ms": voice_duration * 1000,
                        "head_yaw": None,
                        "head_pitch": None,
                        "gaze_vector": None,
                        "confidence_score": min(audio_level * 10, 1.0),
                        "id": pe["id"],
                    })
                    flag = BehaviorFlag.TALKING
                    event_log.append(_event_entry(now, flag, EventSeverity.CRITICAL, "Voice detected via microphone"))
                    new_events.append(event_log[-1])
            else:
                voice_start = 0.0

            response = {
                "face_count": result.face_count,
                "flags": [f.value for f in result.flags],
                "gaze": None,
                "head_pose": None,
                "new_events": new_events,
                "event_log": event_log[-100:],
                "alerts": alerts_to_send,
                "proctoring_events": proctoring_events[-200:],
                "confidence": getattr(result, "confidence", 0),
                "low_light": getattr(result, "low_light", False),
                "talking": getattr(result, "talking", False) or voice_detected,
                "mouth_open_ratio": getattr(result, "mouth_open_ratio", 0.0),
                "voice_detected": voice_detected,
                "audio_level": round(audio_level, 4),
                "doc_verified": doc_verify_failures == 0 and doc_verify_last_time > 0,
                "doc_verify_failures": doc_verify_failures,
                "session_start": session_start,
                "session_student_id": session_student_id,
                "session_student_name": session_student_name,
            }
            if result.gaze:
                response["gaze"] = {
                    "direction": result.gaze.direction.value,
                    "horizontal_ratio": result.gaze.horizontal_ratio,
                    "vertical_ratio": result.gaze.vertical_ratio,
                }
            if result.head_pose:
                response["head_pose"] = {
                    "yaw": result.head_pose.yaw,
                    "pitch": result.head_pose.pitch,
                    "roll": result.head_pose.roll,
                }

            await websocket.send_json(response)

    except WebSocketDisconnect:
        pass
    finally:
        # Clean up session reference photo
        if session_ref_photo_path is not None:
            try:
                session_ref_photo_path.unlink(missing_ok=True)
            except Exception:
                pass
