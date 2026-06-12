# AlgoTrade-X — Launchers

One-click scripts that start the API backend, wait for it to be healthy, then start the Streamlit dashboard and open it in your browser.

| File | Platform | How to run |
|------|----------|------------|
| `start.bat` | Windows 10/11 | Double-click, or `.\launchers\start.bat` in any terminal |
| `start.sh` | Linux & macOS | `bash launchers/start.sh` (or `chmod +x launchers/start.sh && ./launchers/start.sh`) |

## What each launcher does

1. Checks Python is installed (errors out with a clear message if not)
2. Checks `fastapi`, `uvicorn`, `streamlit` are installed — auto-installs if missing
3. Opens the **API backend** (`uvicorn`) in its own terminal window on port **8000**
4. Polls `http://localhost:8000/health` every 2 seconds (up to 120s) until the API responds
5. Opens the **Streamlit dashboard** in its own terminal window on port **8501**
6. Opens `http://localhost:8501` in **Firefox** (falls back to default browser if Firefox isn't found)

## Requirements

- Python 3.10+ in your PATH
- Run from the project root, or from anywhere — the scripts resolve the project directory automatically

## Ports

| Service | URL |
|---------|-----|
| Dashboard | http://localhost:8501 |
| API | http://localhost:8000 |
| API docs (Swagger) | http://localhost:8000/docs |
