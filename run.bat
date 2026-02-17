@echo off
:: ──────────────────────────────────────────────────────────────────────
::  run.bat  —  Start FastAPI + Streamlit (Windows)
:: ──────────────────────────────────────────────────────────────────────
::  Usage:  run.bat
:: ──────────────────────────────────────────────────────────────────────

echo.
echo  🚀  Launching 360° Surveillance System …
echo.

set PYTHONPATH=%~dp0

:: 1. Start FastAPI in the background
echo   ➜  Starting FastAPI server on http://localhost:8000
start "FastAPI Server" /B python -m uvicorn api.server:app --host 0.0.0.0 --port 8000

timeout /t 3 /nobreak >nul

:: 2. Start Streamlit (foreground so you see its logs)
echo   ➜  Starting Streamlit dashboard on http://localhost:8501
streamlit run frontend/dashboard.py --server.port 8501
