"""
api/server.py
─────────────
Headless FastAPI server for the 360° Surveillance Command & Control system.

This is the ONLY process required.  It handles:
  1. AI inference (OpenVINO INT8 via the vision engine).
  2. MJPEG video streaming for all 4 cameras.
  3. JSON REST API for stats, alerts, events, and metadata.
  4. Serving the ultra-lightweight HTML dashboard (no Streamlit).

Endpoints
─────────
GET  /                          → HTML dashboard (single-file SPA)
GET  /api/health                → JSON health check
GET  /video/{cam_id}            → MJPEG stream (multipart/x-mixed-replace)
GET  /snapshot/{cam_id}         → single JPEG frame
GET  /api/stats                 → JSON summary of all cameras
GET  /api/alerts                → JSON list of recent intrusion events
GET  /api/events                → JSON categorised event log
GET  /api/metadata              → compact metadata (intrusion flag)
GET  /api/config/confidence     → current confidence threshold
POST /api/config/confidence     → update confidence threshold

Architecture note
─────────────────
The AnalyticsEngine runs in a background thread.  FastAPI handlers
simply *read* the shared stats dict — no inference happens inside
request handlers, keeping response latency sub-millisecond.
"""

from __future__ import annotations

import os
import time
import asyncio
import logging
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, Response, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config.settings import (
    CAMERAS, CONFIDENCE_THRESHOLD, FASTAPI_HOST, FASTAPI_PORT, JPEG_QUALITY,
)
from backend.vision_engine import CameraManager, AnalyticsEngine

logger = logging.getLogger("api.server")

# ════════════════════════════════════════════════════════════════════
#  Bootstrap the engine once (module-level singletons)
# ════════════════════════════════════════════════════════════════════

cam_manager = CameraManager(CAMERAS)
engine      = AnalyticsEngine(cam_manager, confidence=CONFIDENCE_THRESHOLD)

# ════════════════════════════════════════════════════════════════════
#  FastAPI app
# ════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="360 Surveillance Command Center",
    version="2.0.0",
    description="Edge-optimised INT8 inference + ultra-lightweight HTML dashboard.",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Lifecycle events ─────────────────────────────────────────────────

@app.on_event("startup")
async def _startup():
    cam_manager.start()
    engine.start()
    logger.info("Camera manager + analytics engine started")


@app.on_event("shutdown")
async def _shutdown():
    engine.stop()
    cam_manager.stop()
    logger.info("Engine shut down")


# ════════════════════════════════════════════════════════════════════
#  HTML Dashboard (serves the single-file SPA)
# ════════════════════════════════════════════════════════════════════

DASHBOARD_PATH = Path(__file__).resolve().parent.parent / "frontend" / "index.html"


@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    """Serve the ultra-lightweight HTML/JS dashboard."""
    if DASHBOARD_PATH.exists():
        return HTMLResponse(
            content=DASHBOARD_PATH.read_text(encoding="utf-8"),
            status_code=200,
        )
    return HTMLResponse(
        content="<h1>Dashboard not found</h1><p>Place index.html in frontend/</p>",
        status_code=404,
    )


# ── JSON health check ────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "running", "cameras": len(CAMERAS), "engine": "openvino-int8"}


# ════════════════════════════════════════════════════════════════════
#  MJPEG Streaming (multipart/x-mixed-replace)
# ════════════════════════════════════════════════════════════════════

def _encode_jpeg(frame: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


async def _mjpeg_generator(cam_id: str, fps_cap: int = 15):
    """
    Async generator that yields MJPEG frames.
    fps_cap limits bandwidth without changing inference speed.
    """
    interval = 1.0 / fps_cap
    while True:
        frame = engine.get_annotated_frame(cam_id)
        if frame is not None:
            jpg = _encode_jpeg(frame)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            )
        await asyncio.sleep(interval)


@app.get("/video/{cam_id}")
async def video_feed(cam_id: str, fps: int = Query(15, ge=1, le=30)):
    """Live MJPEG stream for a single camera."""
    valid_ids = [c.cam_id for c in CAMERAS]
    if cam_id not in valid_ids:
        raise HTTPException(404, f"Camera '{cam_id}' not found. Valid: {valid_ids}")
    return StreamingResponse(
        _mjpeg_generator(cam_id, fps),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/snapshot/{cam_id}")
def snapshot(cam_id: str):
    """Return a single JPEG snapshot."""
    frame = engine.get_annotated_frame(cam_id)
    if frame is None:
        raise HTTPException(503, "No frame available yet.")
    jpg = _encode_jpeg(frame, quality=90)
    return Response(content=jpg, media_type="image/jpeg")


# ════════════════════════════════════════════════════════════════════
#  JSON API  (/api/*)
# ════════════════════════════════════════════════════════════════════

@app.get("/api/stats")
def stats():
    """Per-camera detection counts, FPS, and intrusion totals."""
    return JSONResponse(engine.get_stats_snapshot())


@app.get("/api/alerts")
def alerts(n: int = Query(50, ge=1, le=500)):
    """Return the n most recent intrusion / PPE alerts."""
    return JSONResponse(engine.get_alerts(n))


@app.get("/api/events")
def events(n: int = Query(100, ge=1, le=500),
           category: str = Query(None)):
    """
    Return categorised detection events.
    Optional category filter: "Human & PPE", "Vehicle & Plates", "Pet & Animal".
    """
    return JSONResponse(engine.get_events(n, category=category))


@app.get("/api/metadata")
def metadata():
    """
    Compact JSON metadata packet for the dashboard.
    Includes intrusion_active flag for the Red-Alert popup.
    """
    return JSONResponse(engine.get_metadata_packet())


# ── Backward-compatible aliases (old Streamlit routes) ────────────────

@app.get("/stats")
def stats_compat():
    return JSONResponse(engine.get_stats_snapshot())

@app.get("/alerts")
def alerts_compat(n: int = Query(50, ge=1, le=500)):
    return JSONResponse(engine.get_alerts(n))

@app.get("/events")
def events_compat(n: int = Query(100, ge=1, le=500), category: str = Query(None)):
    return JSONResponse(engine.get_events(n, category=category))

@app.get("/metadata")
def metadata_compat():
    return JSONResponse(engine.get_metadata_packet())


# ════════════════════════════════════════════════════════════════════
#  Runtime Configuration
# ════════════════════════════════════════════════════════════════════

class ConfidenceUpdate(BaseModel):
    value: float


@app.post("/api/config/confidence")
def set_confidence(body: ConfidenceUpdate):
    """Adjust the detection confidence threshold (0.05 - 1.0)."""
    engine.confidence = body.value
    return {"confidence": engine.confidence}


@app.get("/api/config/confidence")
def get_confidence():
    return {"confidence": engine.confidence}


# Backward-compatible aliases
@app.post("/config/confidence")
def set_confidence_compat(body: ConfidenceUpdate):
    engine.confidence = body.value
    return {"confidence": engine.confidence}

@app.get("/config/confidence")
def get_confidence_compat():
    return {"confidence": engine.confidence}


# ════════════════════════════════════════════════════════════════════
#  Camera list endpoint (for the HTML dashboard)
# ════════════════════════════════════════════════════════════════════

@app.get("/api/cameras")
def camera_list():
    """Return the list of configured cameras."""
    return JSONResponse([
        {"cam_id": c.cam_id, "label": c.label or c.cam_id, "uri": c.uri}
        for c in CAMERAS
    ])


# ════════════════════════════════════════════════════════════════════
#  Direct-run entry point  (python -m api.server)
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.server:app",
        host=FASTAPI_HOST,
        port=FASTAPI_PORT,
        reload=False,
        log_level="info",
    )
