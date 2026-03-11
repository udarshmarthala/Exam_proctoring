# ProctorAI — AI-Based Online Exam Proctoring System
## Identity Verification Module (Person 1)

A production-ready identity verification system using **LangGraph**, **DeepFace (ArcFace)**, **FastAPI**, and **JWT authentication**.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   FastAPI Backend                    │
│  POST /api/v1/enroll   — Register student face      │
│  POST /api/v1/verify   — Run verification pipeline  │
│  POST /api/v1/auth/validate — Validate JWT token    │
│  GET  /api/v1/students — List enrolled students     │
│  GET  /api/v1/health   — Health check               │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│              LangGraph Agent Pipeline                │
│                                                     │
│  [START]                                            │
│     ↓                                               │
│  [capture]  — Face detection & image validation     │
│     ↓                                               │
│  [liveness] — Anti-spoofing via DeepFace            │
│     ↓                                               │
│  [recognition] — ArcFace embedding comparison       │
│     ↓                                               │
│  [decision] — Fuse scores → approved/rejected/      │
│               escalate                              │
│     ↓                                               │
│  [END]                                              │
└─────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites
- Python 3.9 or higher
- A webcam (optional — photo upload also supported)
- ~2–3GB disk space for DeepFace model weights (downloaded automatically on first run)

### 1. Clone / extract the project
```bash
cd exam_proctor
```

### 2. (Optional) Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate          # Linux / macOS
venv\Scripts\activate             # Windows
```

### 3. Run everything with one command
```bash
python run.py
```

This will:
- Install all dependencies from `requirements.txt`
- Create `.env` from `.env.example` (if `.env` doesn't exist)
- Create required data directories (`data/enrolled`, `data/uploads`)
- Start the FastAPI server on `http://localhost:8000`
- Open the browser automatically

### 4. Manual install (alternative)
```bash
pip install -r requirements.txt
cp .env.example .env        # edit as needed
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Usage

### Browser UI
Open **http://localhost:8000** in your browser.

**Tab 1 — Enroll:**
1. Enter a Student ID (e.g. `STU-001`) and name
2. Upload a clear frontal face photo
3. Optionally upload an ID document
4. Click **Enroll Student**

**Tab 2 — Verify:**
1. Enter the same Student ID
2. Choose Webcam (capture a live frame) or Upload a photo
3. Click **Run Verification**
4. Watch the 4-node LangGraph pipeline animate in real time

**Tab 3 — Result:**
- See **Approved / Rejected / Escalated** verdict with confidence score
- View per-node agent trace (duration, detail, status)
- Copy the JWT token (only issued on approval)
- Validate the token directly in the UI

### API (Swagger UI)
Open **http://localhost:8000/api/docs** for the interactive API documentation.

### Example API calls (curl)

**Enroll:**
```bash
curl -X POST http://localhost:8000/api/v1/enroll \
  -F "student_id=STU-001" \
  -F "student_name=Jane Doe" \
  -F "photo=@/path/to/face.jpg"
```

**Verify:**
```bash
curl -X POST http://localhost:8000/api/v1/verify \
  -F "student_id=STU-001" \
  -F "photo=@/path/to/probe.jpg"
```

**Validate token:**
```bash
curl -X POST http://localhost:8000/api/v1/auth/validate \
  -H "Content-Type: application/json" \
  -d '{"token": "<jwt_token_here>"}'
```

---

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | *(change this!)* | JWT signing key (min 32 chars) |
| `ALGORITHM` | `HS256` | JWT algorithm |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | Token lifetime |
| `DEEPFACE_MODEL` | `ArcFace` | Face recognition model |
| `DEEPFACE_DETECTOR` | `opencv` | Face detector backend |
| `DEEPFACE_DISTANCE_METRIC` | `cosine` | Embedding distance metric |
| `VERIFICATION_THRESHOLD` | `0.40` | Cosine distance threshold |
| `LIVENESS_THRESHOLD` | `0.80` | Anti-spoofing confidence threshold |
| `ENROLLED_FACES_DIR` | `./data/enrolled` | Where enrolled photos are stored |
| `UPLOAD_DIR` | `./data/uploads` | Temporary probe image storage |

---

## Decision Logic

| Condition | Verdict |
|---|---|
| Capture OK + Liveness REAL + Face match (confidence ≥ 55%) | ✅ **Approved** |
| Spoof detected | ❌ **Rejected** |
| No face / multiple faces | ❌ **Rejected** |
| Low confidence (< 30%) | ❌ **Rejected** |
| Liveness UNKNOWN or confidence in 30–55% | ⚠️ **Escalated** |

---

## Project Structure

```
exam_proctor/
├── app/
│   ├── agents/
│   │   ├── graph.py          # LangGraph pipeline (build + compile)
│   │   └── nodes.py          # 4 nodes: capture, liveness, recognition, decision
│   ├── api/
│   │   └── routes.py         # FastAPI endpoints
│   ├── core/
│   │   ├── config.py         # Pydantic settings (reads .env)
│   │   └── security.py       # JWT create/decode utilities
│   ├── models/
│   │   └── schemas.py        # Pydantic request/response models + AgentState
│   └── main.py               # FastAPI app factory
├── static/
│   └── index.html            # Single-file frontend (no framework)
├── data/
│   ├── enrolled/             # Enrolled face photos (created at runtime)
│   └── uploads/              # Temporary probe uploads
├── run.py                    # One-command launcher
├── requirements.txt
├── .env.example
└── README.md
```

---

## Notes

- **First run:** DeepFace will download model weights (~600MB for ArcFace). This happens automatically and only once.
- **GPU support:** If a CUDA-capable GPU is available, DeepFace/TensorFlow will use it automatically.
- **Production deployment:** Change `SECRET_KEY`, set `DEBUG=false`, use a reverse proxy (nginx), and consider using `--workers 4`.
- **Model accuracy:** ArcFace with cosine distance achieves state-of-the-art accuracy on LFW (99.82%). Threshold tuning via `VERIFICATION_THRESHOLD` is recommended for your dataset.

---

## License
MIT — for educational and research purposes.
