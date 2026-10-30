from pathlib import Path
import subprocess, sys, threading, time, urllib.request, webbrowser

root = Path.cwd()
db_file = root / "backend" / "app" / "database.py"
if db_file.exists():
    txt = db_file.read_text(encoding="utf-8")
    if "import tempfile" not in txt:
        txt = txt.replace("import sqlite3", "import sqlite3\nimport tempfile")
    txt = txt.replace(
        'Path("/tmp/multimodal_traffic_platform.db")',
        'Path(tempfile.gettempdir()) / "multimodal_traffic_platform.db"',
    )
    txt = txt.replace(
        "shutil.copyfile(self.db_path, self.runtime_path)",
        "self.runtime_path.parent.mkdir(parents=True, exist_ok=True)\n            shutil.copyfile(self.db_path, self.runtime_path)",
    )
    db_file.write_text(txt, encoding="utf-8")
    print("[OK] Patched backend/app/database.py for Windows 11")

main_file = root / "backend" / "app" / "main.py"
if main_file.exists():
    mtxt = main_file.read_text(encoding="utf-8")
    if "HTMLResponse" not in mtxt:
        mtxt = mtxt.replace(
            "from fastapi.responses import JSONResponse, PlainTextResponse",
            "from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse",
        )
    if '@app.get("/", response_class=HTMLResponse)' not in mtxt:
        route_code = '''

@app.get("/", response_class=HTMLResponse)
@app.get("/index.html", response_class=HTMLResponse)
def serve_frontend_ui() -> HTMLResponse:
    frontend_file = Path(__file__).resolve().parent.parent.parent / "frontend" / "index.html"
    return HTMLResponse(content=frontend_file.read_text(encoding="utf-8"), status_code=200)
'''
        mtxt = mtxt.replace('@app.get("/api/health")', route_code + '\n@app.get("/api/health")')
    main_file.write_text(mtxt, encoding="utf-8")
    print("[OK] Mounted frontend/index.html at http://127.0.0.1:8000/")

print("[STEP 1/3] Running train_and_seed.py...")
subprocess.run([sys.executable, str(root / "backend" / "train_and_seed.py")], cwd=str(root / "backend"), check=True)

print("[STEP 2/3] Running unit and integration test suite...")
subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "backend/tests", "-v"], cwd=str(root), check=True)

def open_when_ready():
    for _ in range(40):
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/api/health", timeout=1.5) as r:
                if r.status == 200:
                    print("\n[READY] Opening http://127.0.0.1:8000 in your default browser...\n")
                    webbrowser.open("http://127.0.0.1:8000", new=2)
                    return
        except Exception:
            time.sleep(0.5)

threading.Thread(target=open_when_ready, daemon=True).start()
print("[STEP 3/3] Starting FastAPI/Uvicorn server on http://127.0.0.1:8000 ...")
subprocess.run([sys.executable, "-m", "uvicorn", "app.main:app", "--app-dir", str(root / "backend"), "--host", "127.0.0.1", "--port", "8000"])
