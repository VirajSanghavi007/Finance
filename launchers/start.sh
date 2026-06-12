#!/usr/bin/env bash
set -euo pipefail

# AlgoTrade-X - Start Backend + Frontend on Linux / macOS.
# Usage: bash launchers/start.sh

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/venv"
VENV_PYTHON="$VENV_DIR/bin/python"
VENV_UVICORN="$VENV_DIR/bin/uvicorn"
VENV_STREAMLIT="$VENV_DIR/bin/streamlit"
API_PORT=8000
DASH_PORT=8501
HEALTH_URL="http://localhost:${API_PORT}/health"
MAX_WAIT=120
POLL_INTERVAL=2

GREEN='\033[0;32m'
AMBER='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

info() { echo -e "${AMBER}[INFO]${NC}  $*"; }
ok() { echo -e "${GREEN}[ OK ]${NC}  $*"; }
err() { echo -e "${RED}[ERR ]${NC}  $*"; exit 1; }

cd "$PROJECT_DIR"

[[ -f "$VENV_PYTHON" ]] || err "Project venv not found at $VENV_PYTHON. Create it with: python3 -m venv venv"

# Activate the venv so subprocesses and console scripts use project-local packages.
source "$VENV_DIR/bin/activate"

info "Using $("$VENV_PYTHON" --version) from project venv"

if ! "$VENV_PYTHON" -c "import fastapi, uvicorn, streamlit" &>/dev/null; then
    info "Installing required packages into project venv..."
    "$VENV_PYTHON" -m pip install -r "$PROJECT_DIR/requirements-dev.txt"
fi

OS="$(uname -s)"

open_terminal() {
    local title="$1"
    local cmd="$2"
    if [[ "$OS" == "Darwin" ]]; then
        osascript -e "tell application \"Terminal\" to do script \"cd '$PROJECT_DIR' && source '$VENV_DIR/bin/activate' && $cmd\""
    elif command -v gnome-terminal &>/dev/null; then
        gnome-terminal --title="$title" -- bash -c "cd '$PROJECT_DIR' && source '$VENV_DIR/bin/activate' && $cmd; exec bash"
    elif command -v konsole &>/dev/null; then
        konsole --title "$title" -e bash -c "cd '$PROJECT_DIR' && source '$VENV_DIR/bin/activate' && $cmd; exec bash" &
    elif command -v xterm &>/dev/null; then
        xterm -title "$title" -e bash -c "cd '$PROJECT_DIR' && source '$VENV_DIR/bin/activate' && $cmd; exec bash" &
    else
        local logfile="$PROJECT_DIR/data/logs/${title// /_}.log"
        mkdir -p "$PROJECT_DIR/data/logs"
        info "No GUI terminal found; running '$title' in background. Log: $logfile"
        bash -c "cd '$PROJECT_DIR' && source '$VENV_DIR/bin/activate' && $cmd" > "$logfile" 2>&1 &
    fi
}

run_uvicorn="$VENV_PYTHON -m uvicorn"
[[ -x "$VENV_UVICORN" ]] && run_uvicorn="$VENV_UVICORN"

run_streamlit="$VENV_PYTHON -m streamlit"
[[ -x "$VENV_STREAMLIT" ]] && run_streamlit="$VENV_STREAMLIT"

info "Starting API backend on port ${API_PORT}..."
open_terminal "AlgoTrade-X API" \
    "$run_uvicorn src.api.main:app --host 0.0.0.0 --port ${API_PORT}"

info "Waiting for API to be ready (max ${MAX_WAIT}s)..."
elapsed=0
until curl -sf "$HEALTH_URL" -o /dev/null 2>/dev/null || \
      "$VENV_PYTHON" -c "import urllib.request; urllib.request.urlopen('$HEALTH_URL', timeout=2)" &>/dev/null; do
    sleep "$POLL_INTERVAL"
    elapsed=$((elapsed + POLL_INTERVAL))
    echo -e "        ... ${elapsed}s / ${MAX_WAIT}s"
    if (( elapsed >= MAX_WAIT )); then
        echo -e "${AMBER}[WARN]${NC}  API not ready after ${MAX_WAIT}s; starting dashboard anyway."
        break
    fi
done
(( elapsed < MAX_WAIT )) && ok "API is healthy after ${elapsed}s."

info "Starting Streamlit dashboard on port ${DASH_PORT}..."
open_terminal "AlgoTrade-X Dashboard" \
    "$run_streamlit run src/dashboard/app.py --server.port ${DASH_PORT} --server.address localhost --server.headless false"

sleep 4
DASH_URL="http://localhost:${DASH_PORT}"
if [[ "$OS" == "Darwin" ]]; then
    if open -Ra "Firefox" &>/dev/null 2>&1; then
        open -a Firefox "$DASH_URL"
    else
        open "$DASH_URL"
    fi
else
    if command -v firefox &>/dev/null; then
        firefox "$DASH_URL" &
    elif command -v xdg-open &>/dev/null; then
        xdg-open "$DASH_URL" &
    fi
fi

echo ""
echo -e "${AMBER}AlgoTrade-X is running${NC}"
echo "API:        http://localhost:${API_PORT}"
echo "Dashboard:  http://localhost:${DASH_PORT}"
echo "API docs:   http://localhost:${API_PORT}/docs"
echo ""
echo "Close the terminal windows to stop."
