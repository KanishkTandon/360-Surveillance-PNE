"""
config/settings.py
──────────────────
Central configuration for the 360° Surveillance System (Edge-Optimised).
Edit this file to add cameras, change model paths, or tune thresholds.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import os
import numpy as np

# ─────────────────────────── Model paths ───────────────────────────
# INT8 OpenVINO IR model exported by  scripts/export_int8.py
YOLO_MODEL_XML = "models/yolov8n_openvino_model/yolov8n.xml"

# ─────────────────────────── OpenVINO Edge Tuning ──────────────────
# Device: "CPU" for pure-CPU edge deployment. "AUTO" to let OV choose.
OV_DEVICE:      str = "CPU"
# Performance hint:
#   "THROUGHPUT"  → maximise total FPS across all 4 cameras (edge default)
#   "LATENCY"     → minimise per-frame latency (single-camera use)
OV_PERF_HINT:   str = "THROUGHPUT"
# Number of inference streams (0 = let OpenVINO auto-select based on cores)
OV_NUM_STREAMS: int = 0
# Pin to a specific thread count (0 = all available logical cores)
OV_NUM_THREADS: int = 0

# ─────────────────────────── Detection classes ─────────────────────
# COCO-80 indices that map to our five target categories.
# Adjust if you retrained the model on a custom dataset.
TARGET_CLASSES: Dict[int, str] = {
    0:  "Person",
    2:  "Vehicle",      # car
    3:  "Vehicle",      # motorcycle
    5:  "Vehicle",      # bus
    7:  "Vehicle",      # truck
    15: "Pet",          # cat
    16: "Pet",          # dog
    # Numberplate & PPE/Helmet require a second-stage model or
    # a custom-trained YOLOv8n. Add class IDs here after retraining.
    # 80: "Numberplate",
    # 81: "Helmet",
}

# Friendly label → emoji for the dashboard
LABEL_ICONS: Dict[str, str] = {
    "Person":      "🧑",
    "Vehicle":     "🚗",
    "Pet":         "🐾",
    "Numberplate": "🔢",
    "Helmet":      "⛑️",
    "Intrusion":   "🚨",
}

# ─────────────────────────── Inference ─────────────────────────────
CONFIDENCE_THRESHOLD: float = 0.45        # default; adjustable from UI
NMS_IOU_THRESHOLD:    float = 0.50
INPUT_SIZE:           Tuple[int, int] = (640, 640)
# AsyncInferQueue depth — rule of thumb: physical_cores / 2
# On a 4-core i5 → 2 slots; on an 8-core i7 → 4 slots
ASYNC_INFER_JOBS:     int = max(2, (os.cpu_count() or 4) // 2)

# ─────────────────────────── Cameras ───────────────────────────────
@dataclass
class CameraSource:
    """Represents one camera feed."""
    cam_id:   str                          # unique key, e.g. "cam_north"
    uri:      str                          # RTSP URL or device index ("0")
    label:    str = ""                     # human-readable name
    roi_poly: List[Tuple[int, int]] = field(default_factory=list)
    # roi_poly defines the intrusion zone as a closed polygon.
    # Leave empty → full-frame detection, no intrusion logic.

# Default demo cameras — swap with your real RTSP URLs
CAMERAS: List[CameraSource] = [
    CameraSource(
        cam_id="cam_north",
        uri="rtsp://192.168.3.35:554/ch01.264",                            # laptop webcam for dev
        label="North Gate",
        roi_poly=[(100, 100), (500, 100), (500, 400), (100, 400)],
    ),
    CameraSource(
        cam_id="cam_south",
        uri="rtsp://192.168.3.5:554/ch01.264",
        label="South Gate",
        roi_poly=[],
    ),
    CameraSource(
        cam_id="cam_east",
        uri="rtsp://192.168.3.12:554/ch01.264",
        label="East Perimeter",
        roi_poly=[(200, 150), (600, 150), (600, 450), (200, 450)],
    ),
    CameraSource(
        cam_id="cam_west",
        uri="0",
        label="West Perimeter",
        roi_poly=[],
    ),
]

# ─────────────────────────── Triage categories ────────────────────
# Maps detection labels to triage columns on the dashboard.
TRIAGE_MAP: Dict[str, str] = {
    "Person":      "Human & PPE",
    "Helmet":      "Human & PPE",
    "Vehicle":     "Vehicle & Plates",
    "Numberplate": "Vehicle & Plates",
    "Pet":         "Pet & Animal",
}

# ─────────────────────────── Shared state / IPC ────────────────────
MAX_ALERT_HISTORY: int = 200               # rolling event log size

# ─────────────────────────── Edge / Camera ─────────────────────────
CAM_RECONNECT_DELAY: float = 0.5           # seconds before camera reconnect
JPEG_QUALITY:        int   = 75            # MJPEG compression (lower = less bandwidth)

# ─────────────────────────── Server ────────────────────────────────
FASTAPI_HOST: str = "0.0.0.0"
FASTAPI_PORT: int = 8000
STREAMLIT_PORT: int = 8501
