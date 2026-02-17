# 360-Surveillance-PNE: Command & Control System Documentation

## 1. Executive Summary

**360-Surveillance-PNE** is an enterprise-grade, real-time computer vision surveillance system designed for complete perimeter security. It leverages state-of-the-art AI (YOLOv8) to detect, classify, and triage security events across multiple camera feeds (North, South, East, West).

The system features a **hybrid inference engine** (OpenVINO with PyTorch fallback), a high-speed **FastAPI backend**, and a responsive **Streamlit Command Center** dashboard. It is engineered for low latency, robust error handling, and ease of deployment on standard hardware.

---

## 2. System Architecture

The solution is built on a decoupled microservices-like architecture:

### 2.1. Backend: The Vision Engine (`backend/vision_engine.py`)
- **Core logic**: Handles camera capture, decoding, inference, and business logic.
- **Hybrid Inference**:
    - **Primary**: OpenVINO AsyncPipe (optimized for Intel CPUs/iGPUs) for maximum FPS.
    - **Fallback**: Ultralytics YOLOv8 (PyTorch) if OpenVINO models are missing.
- **Concurrency**:
    - Dedicated thread per camera for frame capture (latest-frame buffer strategy).
    - Asynchronous inference queue (`AsyncInferQueue`) to saturate hardware utilization.
- **Event Processor**: 
    - **Region of Interest (ROI)**: Polygon-based intrusion detection via `cv2.pointPolygonTest`.
    - **Triage Logic**: Automatically categories detections into "Human & PPE", "Vehicle", or "Pet".

### 2.2. API Layer (`api/server.py`)
- **Framework**: FastAPI (ASGI).
- **Role**: Serves video streams and data to the frontend; acts as the "bridge".
- **Endpoints**:
    - `GET /video/{cam_id}`: Low-latency MJPEG video streams.
    - `GET /events`: Retrieving triage-categorized event logs.
    - `GET /metadata`: Real-time system state (intrusion status, active alerts).
    - `GET /stats`: Performance metrics (FPS per camera, uptime).
- **Communication**: Shared memory state with the Vision Engine (No database required for live session data).

### 2.3. Frontend: Command & Control (`frontend/dashboard.py`)
- **Framework**: Streamlit.
- **Features**:
    - **2x2 Video Grid**: Simultaneous monitoring of 4 feeds.
    - **Red-Alert System**: Visual popup/toast notifications for active intrusions.
    - **Triage Tabs**: Segregated event logs for rapid forensic review.
    - **Dark Mode**: Optimized for 24/7 security operations centers (SOC).

---

## 3. Installation & Setup

### Prerequisites
- **OS**: Windows 10/11 (tested), Linux (Ubuntu 22.04+).
- **Python**: Version 3.8 - 3.11.
- **Hardware**: Intel Core i5/i7 (8th Gen+) recommended for OpenVINO optimization. Discrete GPU optional.

### Step-by-Step Deployment

1. **Clone the Repository**
   ```bash
   git clone https://github.com/KanishkTandon/360-Surveillance-PNE.git
   cd 360-Surveillance-PNE
   ```

2. **Environment Setup**
   Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: This installs `ultralytics`, `openvino`, `fastapi`, `streamlit`, `opencv-python`, etc.*

4. **Model Export (Critical for Performance)**
   To enable the high-performance OpenVINO engine:
   ```bash
   yolo export model=yolov8n.pt format=openvino half=True
   ```
   This generates the optimized `.xml` and `.bin` files in `models/yolov8n_openvino_model/`.

---

## 4. Configuration Manual

The system is configured via `config/settings.py` and `config/zones.yaml`.

### 4.1. Camera Setup (`config/settings.py`)
Modify the `CAMERAS` list to define your sources. Supported formats: **RTSP**, **HTTP**, or **USB Index** (0, 1).

```python
CAMERAS = [
    CameraSource(
        cam_id="cam_north",
        uri="rtsp://admin:password@192.168.1.10:554/stream1",
        label="North Gate",
        roi_poly=[(100, 100), (500, 100), (500, 400), (100, 400)]  # Intrusion Zone
    ),
    # ... add up to 4 cameras
]
```

### 4.2. Detection Classes
The `TARGET_CLASSES` dictionary maps COCO model IDs to system labels.
- `0`: Person
- `2, 3, 5, 7`: Vehicles (Car, Motorcycle, Bus, Truck)
- `15, 16`: Pets (Cat, Dog)

### 4.3. Triage Mapping
`TRIAGE_MAP` controls which dashboard tab an event appears in:
- **Human & PPE**: Person, Helmet usually goes here.
- **Vehicle & Plates**: Car, Trust, License Plates.
- **Pet & Animal**: Cats, Dogs, Wildlife.

---

## 5. Usage Guide

### Starting the System
Use the provided launch scripts for a one-click start.

- **Windows**: Double-click `run.bat` (or `run.ps1` in PowerShell).
- **Linux/Mac**: Run `./run.sh`.

**Manual Startup**:
1. **Terminal 1 (Backend)**:
   ```bash
   python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
   ```
2. **Terminal 2 (Frontend)**:
   ```bash
   python -m streamlit run frontend/dashboard.py --server.port 8501
   ```

### Dashboard Navigation
1. **Live View**: The main 2x2 grid shows real-time feeds.
   - **Green Bounding Box**: Safe detection.
   - **Red Bounding Box**: Intrusion detected (Object inside ROI).
2. **Intrusion Alerts**:
   - When an intrusion occurs, a **Red Banner** appears at the top.
   - Click the **"Acknowledge & Dismiss"** button to clear the alert state.
3. **Sidebar**:
   - **Confidence Threshold**: Adjust the slider (0.0 - 1.0) to filter weak detections.
4. **Triage Tabs** (Bottom):
   - Switch between tabs to view historical logs filtered by category.

---

## 6. Directory Structure

```
360-Surveillance-PNE/
├── api/
│   ├── server.py           # FastAPI application & endpoints
├── backend/
│   ├── vision_engine.py    # Core AI logic & OpenVINO integration
├── config/
│   ├── settings.py         # Main configuration file
│   ├── zones.yaml          # (Optional) External zone definitions
├── frontend/
│   ├── dashboard.py        # Streamlit user interface
├── models/                 # Directory for .pt and OpenVINO .xml/.bin files
├── .streamlit/             # Streamlit theme capability
├── run.bat                 # Windows Launcher
├── requirements.txt        # Python dependency list
└── DOCUMENTATION.md        # This file
```

## 7. Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **"Could not open file... .xml"** | OpenVINO model not exported. | Run `yolo export model=yolov8n.pt format=openvino half=True`. System will use fallback until fixed. |
| **Stream Lag / Delayed Video** | RTSP decoding bottleneck. | Use OpenVINO model (CPU accel). Ensure network bandwidth is sufficient. |
| **"Intrusion" not triggering** | Object logic center not in ROI. | Adjust `roi_poly` coordinates in variable `settings.py`. Ensure points are `(x, y)`. |
| **Camera not loading** | Invalid RTSP URI or network down. | Check camera IP ping. Verify URI in VLC Player. |

---

**© 2026 Kanishk Tandon. All Rights Reserved.**
