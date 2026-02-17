# 🔒 360° Street Surveillance System

**YOLOv8n + OpenVINO + FastAPI + Streamlit** — a modular, high-performance edge computer-vision system with a "Glass Box" dashboard.

---

## 📐 Architecture

```
┌───────────────────────────────────────────────────────────┐
│                    STREAMLIT DASHBOARD                    │
│   (frontend/dashboard.py — port 8501)                    │
│                                                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    │
│  │ Cam North│ │ Cam South│ │ Cam East │ │ Cam West │    │
│  │  MJPEG   │ │  MJPEG   │ │  MJPEG   │ │  MJPEG   │    │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘    │
│       │             │             │             │          │
│  ┌────┴─────────────┴─────────────┴─────────────┴────┐   │
│  │          HTTP GET /video/{cam_id}  (MJPEG)        │   │
│  │          HTTP GET /stats           (JSON)         │   │
│  │          HTTP GET /alerts          (JSON)         │   │
│  └───────────────────┬───────────────────────────────┘   │
└───────────────────────┼───────────────────────────────────┘
                        │
         ┌──────────────▼──────────────┐
         │       FASTAPI SERVER        │
         │   (api/server.py — port 8000)│
         │                              │
         │  Reads from shared state:    │
         │  • annotated frames (dict)   │
         │  • stats (dict)              │
         │  • alerts (deque)            │
         └──────────────┬──────────────┘
                        │  in-process (same Python)
         ┌──────────────▼──────────────┐
         │     ANALYTICS ENGINE        │
         │  (backend/vision_engine.py) │
         │                              │
         │  ┌────────────────────────┐  │
         │  │  OpenVINOInference     │  │
         │  │  AsyncInferQueue (x4)  │  │
         │  └────────────────────────┘  │
         │  ┌────────────────────────┐  │
         │  │  CameraManager         │  │
         │  │  1 thread per camera   │  │
         │  └────────────────────────┘  │
         │  ┌────────────────────────┐  │
         │  │  Intrusion Detection   │  │
         │  │  cv2.pointPolygonTest  │  │
         │  └────────────────────────┘  │
         └─────────────────────────────┘
```

### How Concurrency Works

| Component | Mechanism | Why |
|---|---|---|
| **Camera capture** | 1 daemon thread per camera → single-slot buffer | Prevents frame accumulation lag |
| **Inference** | `AsyncInferQueue` (N=4 parallel infer requests) | Saturates the OpenVINO device |
| **Analytics loop** | 1 daemon thread iterating over all cameras | Keeps processing off the API thread |
| **FastAPI** | Reads shared `dict` + `deque` (thread-locked) | Sub-ms response latency |
| **Streamlit** | Separate process, polls FastAPI every 2 s | UI never blocks the AI engine |

> **Scaling to Redis / multi-node:** Replace the in-memory `stats` dict and `alerts` deque with a Redis pub/sub channel. The API endpoints stay identical — only the data source changes.

---

## 🎯 Features

| Feature | Status |
|---|---|
| YOLOv8n object detection (People, Vehicles, Pets) | ✅ |
| OpenVINO IR acceleration with AsyncInferQueue | ✅ |
| Ultralytics fallback (no OpenVINO required) | ✅ |
| Polygon ROI intrusion detection | ✅ |
| MJPEG live streams via FastAPI | ✅ |
| JSON stats & alerts endpoints | ✅ |
| Streamlit multi-camera grid | ✅ |
| Live Alerts sidebar | ✅ |
| Confidence threshold slider (runtime) | ✅ |
| Per-camera FPS display | ✅ |
| Event log table | ✅ |
| Numberplate detection | 🔧 (requires custom model) |
| PPE / Helmet detection | 🔧 (requires custom model) |

---

## 📂 Project Structure

```
360-surveillance/
├── config/
│   ├── __init__.py
│   └── settings.py          # cameras, model paths, thresholds
├── backend/
│   ├── __init__.py
│   └── vision_engine.py     # OpenVINOInference, CameraManager, AnalyticsEngine
├── api/
│   ├── __init__.py
│   └── server.py            # FastAPI: MJPEG streams, /stats, /alerts
├── frontend/
│   ├── __init__.py
│   └── dashboard.py         # Streamlit Glass Box dashboard
├── models/                  # place .xml/.bin OpenVINO IR files here
├── requirements.txt
├── run.ps1                  # Windows launcher
├── run.sh                   # Linux/macOS launcher
└── README.md
```

---

## 🚀 Launch Guide

### 1. Prerequisites

- **Python 3.10+**
- A webcam (device `0`) or RTSP camera URLs
- (Optional) OpenVINO for hardware-accelerated inference

### 2. Install Dependencies

```bash
# Create & activate a virtual environment (recommended)
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# Install all packages
pip install -r requirements.txt
```

### 3. Export YOLOv8n to OpenVINO IR (optional)

```bash
# This creates models/yolov8n_openvino_model/yolov8n.xml + .bin
yolo export model=yolov8n.pt format=openvino
```

If you skip this step, the system automatically falls back to Ultralytics
PyTorch inference.

### 4. Configure Cameras

Edit `config/settings.py` → `CAMERAS` list. Replace `uri="0"` with your
RTSP URLs:

```python
CameraSource(
    cam_id="cam_north",
    uri="rtsp://admin:pass@192.168.1.100:554/stream1",
    label="North Gate",
    roi_poly=[(100, 100), (500, 100), (500, 400), (100, 400)],
),
```

### 5. Run the System

#### Option A: One-click launcher

```powershell
# Windows
.\run.ps1
```

```bash
# Linux / macOS
chmod +x run.sh && ./run.sh
```

#### Option B: Two terminals (manual)

**Terminal 1 — FastAPI backend:**
```bash
# Set PYTHONPATH so modules resolve correctly
# Windows PowerShell:
$env:PYTHONPATH = (Get-Location).Path

# Linux / macOS:
export PYTHONPATH=$(pwd)

python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

**Terminal 2 — Streamlit frontend:**
```bash
# Same PYTHONPATH as above
streamlit run frontend/dashboard.py --server.port 8501
```

### 6. Open the Dashboard

Navigate to **http://localhost:8501** in your browser.

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/video/{cam_id}?fps=15` | MJPEG stream |
| `GET` | `/snapshot/{cam_id}` | Single JPEG |
| `GET` | `/stats` | Per-camera JSON metrics |
| `GET` | `/alerts?n=50` | Recent alert events |
| `GET` | `/config/confidence` | Current threshold |
| `POST` | `/config/confidence` | Update threshold `{"value": 0.5}` |

---

## 🧠 Technical Notes

### OpenVINO AsyncInferQueue

The `AsyncInferQueue` allows N inference requests to be in-flight
simultaneously on the same compiled model. This is critical for multi-camera
setups — while one request waits for the VPU/GPU, another can start, keeping
hardware utilisation near 100 %.

```python
queue = AsyncInferQueue(compiled_model, jobs=4)
queue.set_callback(on_done)
queue.start_async({input_name: tensor}, userdata=cam_id)
queue.wait_all()
```

### Intrusion Detection Geometry

Each camera can define a polygon ROI in `config/settings.py`. The
`AnalyticsEngine` uses `cv2.pointPolygonTest` on the centre of every
detected bounding box. If the point is **inside** the polygon the detection
is flagged as an intrusion and pushed to the alert log.

### Inter-Process Communication

For a single-machine deployment, the backend and API run in the **same
Python process** — shared memory via a thread-locked `dict` and `deque`
is the fastest and simplest IPC. Streamlit runs as a **separate process**
and communicates over HTTP.

For multi-node / high-availability setups, swap the in-memory state with:
- **Redis** (pub/sub + streams)
- **ZeroMQ** (for low-latency frame transport)
- **Apache Kafka** (for durable event logs)

---

## 📜 License

MIT — use freely for commercial and personal projects.
