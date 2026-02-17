"""
config/settings.py
──────────────────
Central configuration for the 360° Surveillance System.
Edit this file to add cameras, change model paths, or tune thresholds.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np

# ─────────────────────────── Model paths ───────────────────────────
# Place your OpenVINO IR files (.xml + .bin) inside the models/ folder.
# The default expects a YOLOv8n model exported via:
#   yolo export model=yolov8n.pt format=openvino
YOLO_MODEL_XML = "models/yolov8n_openvino_model/yolov8n.xml"

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
ASYNC_INFER_JOBS:     int = 4             # AsyncInferQueue depth

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
        uri="0",                            # laptop webcam for dev
        label="North Gate",
        roi_poly=[(100, 100), (500, 100), (500, 400), (100, 400)],
    ),
    CameraSource(
        cam_id="cam_south",
        uri="0",
        label="South Gate",
        roi_poly=[],
    ),
    CameraSource(
        cam_id="cam_east",
        uri="0",
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

# ─────────────────────────── Server ────────────────────────────────
FASTAPI_HOST: str = "0.0.0.0"
FASTAPI_PORT: int = 8000
STREAMLIT_PORT: int = 8501
