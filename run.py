#!/usr/bin/env python3
"""
run.py — Single-command launcher for the AI Exam Proctoring System.

Usage:
    python run.py
    python run.py --host 0.0.0.0 --port 8000
    python run.py --reload          (development mode with auto-reload)

The script:
  1. Checks & installs any missing dependencies
  2. Creates required data directories
  3. Copies .env.example → .env if .env doesn't exist
  4. Starts the FastAPI/Uvicorn server
  5. Opens the browser automatically
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


BASE_DIR = Path(__file__).parent


def print_banner():
    print("\n" + "═" * 62)
    print("  ██████╗ ██████╗  ██████╗  ██████╗████████╗ ██████╗ ██████╗ ")
    print("  ██╔══██╗██╔══██╗██╔═══██╗██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗")
    print("  ██████╔╝██████╔╝██║   ██║██║        ██║   ██║   ██║██████╔╝")
    print("  ██╔═══╝ ██╔══██╗██║   ██║██║        ██║   ██║   ██║██╔══██╗")
    print("  ██║     ██║  ██║╚██████╔╝╚██████╗   ██║   ╚██████╔╝██║  ██║")
    print("  ╚═╝     ╚═╝  ╚═╝ ╚═════╝  ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝")
    print()
    print("  AI Exam Proctoring — Identity Verification Module  v1.0.0")
    print("  LangGraph · DeepFace (ArcFace) · FastAPI · JWT")
    print("═" * 62 + "\n")


def check_python_version():
    if sys.version_info < (3, 9):
        print("[ERROR] Python 3.9+ is required. You have:", sys.version)
        sys.exit(1)
    print(f"[✓] Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")


def install_dependencies():
    req_file = BASE_DIR / "requirements.txt"
    if not req_file.exists():
        print("[WARN] requirements.txt not found — skipping install.")
        return

    print("[·] Checking / installing dependencies (this may take a few minutes the first time)…")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(req_file), "--quiet"],
        capture_output=False,
    )
    if result.returncode != 0:
        print("[WARN] Some packages may have failed to install. Check output above.")
    else:
        print("[✓] Dependencies ready")


def setup_env():
    env_file = BASE_DIR / ".env"
    example  = BASE_DIR / ".env.example"
    if not env_file.exists() and example.exists():
        shutil.copy(example, env_file)
        print("[✓] Created .env from .env.example (edit it to customise settings)")
    elif env_file.exists():
        print("[✓] .env file found")


def create_directories():
    dirs = [
        BASE_DIR / "data" / "enrolled",
        BASE_DIR / "data" / "uploads",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print("[✓] Data directories ready")


def open_browser(host: str, port: int, delay: float = 2.0):
    url = f"http://{'localhost' if host == '0.0.0.0' else host}:{port}"
    def _open():
        time.sleep(delay)
        webbrowser.open(url)
    import threading
    t = threading.Thread(target=_open, daemon=True)
    t.start()


def run_server(host: str, port: int, reload: bool, workers: int):
    print(f"\n[✓] Starting server at http://{host}:{port}")
    print(f"    Frontend : http://localhost:{port}/")
    print(f"    API docs : http://localhost:{port}/api/docs")
    print(f"    Press CTRL+C to stop\n")

    cmd = [
        sys.executable, "-m", "uvicorn",
        "app.main:app",
        "--host", host,
        "--port", str(port),
        "--log-level", "info",
    ]
    if reload:
        cmd.append("--reload")
    else:
        cmd += ["--workers", str(workers)]

    os.chdir(BASE_DIR)
    subprocess.run(cmd)


def parse_args():
    p = argparse.ArgumentParser(description="ProctorAI — Identity Verification Launcher")
    p.add_argument("--host",    default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    p.add_argument("--port",    default=8000, type=int, help="Bind port (default: 8000)")
    p.add_argument("--reload",  action="store_true", help="Enable auto-reload (dev mode)")
    p.add_argument("--workers", default=1, type=int, help="Worker count (default: 1)")
    p.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    p.add_argument("--skip-install", action="store_true", help="Skip pip install step")
    return p.parse_args()


def main():
    print_banner()
    args = parse_args()

    check_python_version()
    if not args.skip_install:
        install_dependencies()
    setup_env()
    create_directories()

    if not args.no_browser:
        open_browser(args.host, args.port, delay=3.0)

    run_server(args.host, args.port, args.reload, args.workers)


if __name__ == "__main__":
    main()
