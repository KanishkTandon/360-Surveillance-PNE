# System Architecture — INT8 Edge Pipeline

```mermaid
flowchart TB
    subgraph EXPORT["📦 Model Export (One-Time)"]
        PT["yolov8n.pt<br/>(FP32 PyTorch)"]
        NNCF["NNCF PTQ<br/>Calibration"]
        INT8["yolov8n.xml + .bin<br/>(INT8 OpenVINO IR)"]
        PT --> NNCF --> INT8
    end

    subgraph CAMERAS["📹 Camera Feeds (4x RTSP/USB)"]
        C1["cam_north<br/>North Gate"]
        C2["cam_south<br/>South Gate"]
        C3["cam_east<br/>East Perimeter"]
        C4["cam_west<br/>West Perimeter"]
    end

    subgraph ENGINE["🧠 Vision Engine (CPU-Only)"]
        direction TB

        subgraph CAPTURE["Threaded Capture"]
            T1["Thread 1"]
            T2["Thread 2"]
            T3["Thread 3"]
            T4["Thread 4"]
        end

        subgraph PREPROCESS["HW-Accelerated Preprocessing"]
            PPP["ov.preprocess<br/>• u8→f32 conversion<br/>• Bilinear resize 640×640<br/>• /255.0 normalization<br/>• NHWC→NCHW transpose<br/><i>Runs on SIMD, not Python</i>"]
        end

        subgraph INFERENCE["INT8 AsyncInferQueue"]
            SLOT1["Slot 1"]
            SLOT2["Slot 2"]
            SLOT3["Slot N<br/>(cores/2)"]
            VNNI["AVX-512 VNNI / AMX<br/>INT8 Matrix Multiply"]
        end

        subgraph POSTPROCESS["Post-Processing"]
            NMS["NMS + Decode<br/>(NumPy + cv2.dnn)"]
            ROI["ROI Intrusion Check<br/>(cv2.pointPolygonTest)"]
            TRIAGE["Event Triage<br/>Human│Vehicle│Pet"]
        end

        CAPTURE --> PPP --> INFERENCE
        INFERENCE --> VNNI
        VNNI --> POSTPROCESS
    end

    subgraph API["⚡ FastAPI Server (Port 8000)"]
        direction LR
        MJPEG["/video/{cam_id}<br/>MJPEG Stream"]
        JSON_STATS["/api/stats<br/>JSON Metrics"]
        JSON_EVENTS["/api/events<br/>Triage Events"]
        JSON_META["/api/metadata<br/>Intrusion Flag"]
        HTML_DASH["/ (root)<br/>HTML Dashboard"]
    end

    subgraph BROWSER["🖥️ Browser (Ultra-Lightweight)"]
        direction TB
        GRID["2×2 Camera Grid<br/>(MJPEG <img> tags)"]
        METRICS_UI["Metric Cards<br/>(fetch → DOM)"]
        ALERT_UI["Red-Alert Banner<br/>(Vanilla JS)"]
        TABS_UI["Triage Tabs<br/>(fetch → table)"]
    end

    C1 --> T1
    C2 --> T2
    C3 --> T3
    C4 --> T4

    INT8 -.->|"Loaded at startup"| INFERENCE

    POSTPROCESS -->|"Shared Memory<br/>(thread-locked dict)"| API

    MJPEG -->|"multipart/x-mixed-replace"| GRID
    JSON_STATS -->|"fetch() every 2s"| METRICS_UI
    JSON_META -->|"fetch() every 2s"| ALERT_UI
    JSON_EVENTS -->|"fetch() every 2s"| TABS_UI
    HTML_DASH -->|"Single HTML file"| BROWSER

    style EXPORT fill:#1a2332,stroke:#58a6ff,color:#e6edf3
    style ENGINE fill:#161b22,stroke:#3fb950,color:#e6edf3
    style API fill:#1a2332,stroke:#d29922,color:#e6edf3
    style BROWSER fill:#161b22,stroke:#58a6ff,color:#e6edf3
    style CAMERAS fill:#1a2332,stroke:#8b949e,color:#e6edf3
    style INT8 fill:#0d4429,stroke:#3fb950,color:#e6edf3
    style VNNI fill:#0d4429,stroke:#3fb950,color:#e6edf3
```

## Data Flow Summary

```
Camera (BGR u8) ──► ov.preprocess (SIMD) ──► INT8 Inference (AVX-512/AMX)
                                                      │
                                                      ▼
                                              NMS + ROI Check
                                                      │
                                    ┌─────────────────┼──────────────────┐
                                    ▼                 ▼                  ▼
                              Annotated          JSON Stats        Triage Events
                              Frame (JPEG)       + Alerts          + Intrusions
                                    │                 │                  │
                                    ▼                 ▼                  ▼
                              MJPEG Stream      /api/stats         /api/events
                              (/video/*)        /api/metadata      /api/alerts
                                    │                 │                  │
                                    └─────────┬───────┘──────────────────┘
                                              ▼
                                    Browser (index.html)
                                    • <img> tags for video
                                    • fetch() for JSON
                                    • Vanilla JS DOM updates
```

## INT8 Performance Advantages

| Metric | FP32 | INT8 | Improvement |
|--------|------|------|-------------|
| Model Size | ~24 MB | ~6 MB | 4× smaller |
| Cache Fit | L3 only | L2 + L1 | Better locality |
| SIMD Width | 16 ops/cycle | 64 ops/cycle (VNNI) | 4× throughput |
| Typical FPS (i7-12th) | 15-20 FPS | 45-60 FPS | 3× faster |
| Power (TDP) | ~35W | ~20W | 40% lower |
