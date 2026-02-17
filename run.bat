@echo off
:: ──────────────────────────────────────────────────────────────────────
::  run.bat  —  Start 360° Surveillance Edge System (Windows)
:: ──────────────────────────────────────────────────────────────────────
::  The HTML dashboard is served directly by FastAPI at http://localhost:8000
::  No Streamlit required — single process, minimal resource usage.
:: ──────────────────────────────────────────────────────────────────────

echo.
echo  ==============================================
echo    360 Surveillance — Edge Mode (INT8 / CPU)
echo  ==============================================
echo.

set PYTHONPATH=%~dp0
set PYTHONIOENCODING=utf-8

:: Check if INT8 model exists
if not exist "models\yolov8n_openvino_model\yolov8n.xml" (
    echo   [!] INT8 model not found. Exporting now...
    python scripts\export_int8.py
    echo.
)

:: Start FastAPI (serves both API + HTML dashboard)
echo   Starting server on http://localhost:8000
echo   Dashboard:  http://localhost:8000
echo   API docs:   http://localhost:8000/docs
echo.
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
