#!/usr/bin/env python3
"""
One-Command Cross-Platform Launcher for the Multimodal Traffic Incident Detection
and Road Network Recovery Time Prediction Platform.

Capabilities:
1. Verifies Python runtime and required packages (fastapi, uvicorn, numpy, pandas, cv2, pydantic).
2. Verifies SQLite database & platform_bundle.json exist; automatically runs train_and_seed.py if missing.
3. Checks if the target port (default 8000) is already serving /api/health to prevent port conflicts or duplicate launches.
4. Starts the FastAPI / Uvicorn server in a managed subprocess or foreground process.
5. Polls /api/health until HTTP 200 OK is returned, then opens the default browser once at http://127.0.0.1:8000.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
DB_FILE = BACKEND_DIR / "data" / "traffic_platform.db"
BUNDLE_FILE = BACKEND_DIR / "data" / "platform_bundle.json"

REQUIRED_MODULES = [
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("cv2", "opencv-python"),
    ("pydantic", "pydantic"),
]


def check_dependencies() -> bool:
    missing = []
    for mod_name, pkg_name in REQUIRED_MODULES:
        try:
            importlib.import_module(mod_name)
        except ImportError:
            missing.append(pkg_name)
    if missing:
        print(f"[ERROR] Missing required Python packages: {', '.join(missing)}")
        print(f"        Run: {sys.executable} -m pip install -r requirements.txt")
        return False
    return True


def ensure_seeded() -> None:
    if not DB_FILE.exists() or not BUNDLE_FILE.exists():
        print("[INIT] Database or model bundle missing. Running train_and_seed.py...")
        subprocess.run(
            [sys.executable, str(BACKEND_DIR / "train_and_seed.py")],
            cwd=str(BACKEND_DIR),
            check=True,
        )


def check_health_endpoint(host: str, port: int, timeout: float = 2.0) -> bool:
    url = f"http://{host}:{port}/api/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("status") == "healthy"
    except Exception:
        return False
    return False


def is_port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def poll_and_open_browser(host: str, port: int, open_browser: bool) -> None:
    url = f"http://{host}:{port}"
    for _ in range(60):
        if check_health_endpoint(host, port, timeout=1.5):
            print(f"\n[READY] Backend health check passed: {url}/api/health -> HTTP 200 OK")
            print(f"[READY] Web Application Live URL:    {url}\n")
            if open_browser:
                try:
                    webbrowser.open(url, new=2)
                except Exception as exc:
                    print(f"[WARN] Could not auto-open browser ({exc}). Open {url} manually.")
            return
        time.sleep(0.5)
    print(f"[WARN] Server startup timed out waiting for {url}/api/health.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Start Multimodal Traffic Incident Platform")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the browser")
    parser.add_argument("--check-only", action="store_true", help="Verify dependencies and DB readiness then exit")
    args = parser.parse_args()

    print("=" * 74)
    print(" MULTIMODAL TRAFFIC INCIDENT DETECTION & RECOVERY PREDICTION PLATFORM")
    print("=" * 74)

    if not check_dependencies():
        return 1

    ensure_seeded()

    if args.check_only:
        print("[OK] Environment dependencies and SQLite database verified.")
        return 0

    auto_open = not args.no_browser and os.environ.get("AUTO_OPEN_BROWSER", "true").lower() != "false"

    # If already running and healthy on target port, do not crash with port conflict
    if check_health_endpoint(args.host, args.port):
        url = f"http://{args.host}:{args.port}"
        print(f"[INFO] Platform is already running and healthy at {url}")
        if auto_open:
            webbrowser.open(url, new=2)
        return 0

    if is_port_in_use(args.host, args.port):
        print(f"[ERROR] Port {args.port} is currently occupied by another process.")
        print(f"        Run stop_platform.ps1 or specify a different port: python run_platform.py --port 8001")
        return 1

    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))

    watcher = threading.Thread(
        target=poll_and_open_browser,
        args=(args.host, args.port, auto_open),
        daemon=True,
    )
    watcher.start()

    import uvicorn
    print(f"[START] Starting FastAPI/Uvicorn server on http://{args.host}:{args.port} ...")
    uvicorn.run("app.main:app", host=args.host, port=args.port, app_dir=str(BACKEND_DIR), log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
