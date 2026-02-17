#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
#  run.sh  —  Start FastAPI + Streamlit (Linux / macOS)
# ──────────────────────────────────────────────────────────────────────
#  Usage:  chmod +x run.sh && ./run.sh
# ──────────────────────────────────────────────────────────────────────

set -e

echo ""
echo "🚀  Launching 360° Surveillance System …"
echo ""

export PYTHONPATH="$(pwd)"

# 1. Start FastAPI in the background
echo "  ➜  Starting FastAPI server on http://localhost:8000"
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!

sleep 3   # give the backend a moment

# 2. Start Streamlit (foreground)
echo "  ➜  Starting Streamlit dashboard on http://localhost:8501"
streamlit run frontend/dashboard.py --server.port 8501

# Cleanup on exit
trap "kill $FASTAPI_PID 2>/dev/null" EXIT
