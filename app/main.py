"""
FastAPI application entry point for the AI Exam Proctoring — Identity Verification Module.
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import settings

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Silence noisy third-party debug loggers
for _noisy in ("python_multipart", "python_multipart.multipart", "urllib3", "h5py", "tensorflow", "PIL"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

app = FastAPI(
    title="AI Exam Proctoring — Identity Verification",
    description=(
        "LangGraph-powered identity verification system using DeepFace (ArcFace) "
        "for face recognition and anti-spoofing."
    ),
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(router, prefix="/api/v1")

# Serve static frontend
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
    logger.info("Serving static frontend from %s", static_dir)
else:
    logger.warning("Static directory not found at %s", static_dir)


@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("  AI Exam Proctoring System — Identity Verification")
    logger.info("  Server : http://%s:%s", settings.HOST, settings.PORT)
    logger.info("  API    : http://%s:%s/api/docs", settings.HOST, settings.PORT)
    logger.info("=" * 60)
