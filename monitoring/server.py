#!/usr/bin/env python3
"""
Standalone monitoring server (for development/testing).

In production, the monitoring routes are integrated into the main app (app/main.py).
To run standalone: python -m monitoring.server
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from monitoring.routes import router

app = FastAPI(title="Monitoring Test (Standalone)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
async def health():
    return {"status": "ok", "message": "Standalone monitoring server"}


if __name__ == "__main__":
    port = 8501
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    print(f"Starting monitoring server at http://127.0.0.1:{port}")
    print(f"  Open in browser: http://localhost:{port}/monitoring/exam")
    print(f"  Health check:    http://localhost:{port}/health")
    uvicorn.run(app, host="0.0.0.0", port=port)
