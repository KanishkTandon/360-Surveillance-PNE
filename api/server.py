"""
api/server.py
─────────────
FastAPI bridge between the CV backend and the Streamlit frontend.

Endpoints
---------
GET  /                          → health check
GET  /video/{cam_id}            → MJPEG stream (multipart/x-mixed-replace)
GET  /snapshot/{cam_id}         → single JPEG frame
GET  /stats                     → JSON summary of all cameras
GET  /alerts                    → JSON list of recent intrusion events
POST /config/confidence         → update confidence threshold at runtime

Architecture note
-----------------
The ``AnalyticsEngine`` is instantiated once at startup and runs its
processing loop in a background thread.  FastAPI handlers simply *read*
the shared ``stats`` dict and ``alerts`` deque — no inference happens
inside request handlers, so latency stays sub-millisecond.
"""

from __future__ import annotations

import io
import time
import asyncio
import logging

import cv2
import numpy as np
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config.settings import CAMERAS, CONFIDENCE_THRESHOLD, FASTAPI_HOST, FASTAPI_PORT
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
    title="360° Surveillance API",
    version="1.0.0",
    description="MJPEG streams and live stats for the Glass Box dashboard.",
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


# ── Health check ─────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "running", "cameras": len(CAMERAS)}


# ════════════════════════════════════════════════════════════════════
#  MJPEG Streaming (multipart/x-mixed-replace)
# ════════════════════════════════════════════════════════════════════

def _encode_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


async def _mjpeg_generator(cam_id: str, fps_cap: int = 15):
    """
    Async generator that yields MJPEG frames.
    ``fps_cap`` limits bandwidth without changing inference speed.
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
#  JSON Stats & Alerts
# ════════════════════════════════════════════════════════════════════

@app.get("/stats")
def stats():
    """
    Returns per-camera detection counts, FPS, and intrusion totals.
    Example response:
    {
      "cam_north": {"cam_label": "North Gate", "fps": 18.3,
                    "detections": {"Person": 3, "Vehicle": 1},
                    "intrusions": 1}
    }
    """
    return JSONResponse(engine.get_stats_snapshot())


@app.get("/alerts")
def alerts(n: int = Query(50, ge=1, le=500)):
    """Return the *n* most recent intrusion / PPE alerts."""
    return JSONResponse(engine.get_alerts(n))


# ════════════════════════════════════════════════════════════════════
#  Categorised Events (Triage view)
# ════════════════════════════════════════════════════════════════════

@app.get("/events")
def events(n: int = Query(100, ge=1, le=500),
           category: str = Query(None)):
    """
    Return categorised detection events.
    Optional ``category`` filter: "Human & PPE", "Vehicle & Plates", "Pet & Animal".
    """
    return JSONResponse(engine.get_events(n, category=category))


@app.get("/metadata")
def metadata():
    """
    Compact JSON metadata packet for the dashboard.
    Includes ``intrusion_active`` flag so the frontend can trigger
    the Red-Alert popup without parsing the full event stream.
    """
    return JSONResponse(engine.get_metadata_packet())


# ════════════════════════════════════════════════════════════════════
#  Runtime Configuration
# ════════════════════════════════════════════════════════════════════

class ConfidenceUpdate(BaseModel):
    value: float


@app.post("/config/confidence")
def set_confidence(body: ConfidenceUpdate):
    """Adjust the detection confidence threshold (0.05 – 1.0)."""
    engine.confidence = body.value
    return {"confidence": engine.confidence}


@app.get("/config/confidence")
def get_confidence():
    return {"confidence": engine.confidence}


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
