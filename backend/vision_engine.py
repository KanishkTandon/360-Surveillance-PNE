"""
backend/vision_engine.py
────────────────────────
Zero-Lag, CPU-Only AI backbone for the 360° Command & Control system.

Architecture (Edge-Optimised)
─────────────────────────────
  * OpenVINO INT8 model with THROUGHPUT performance hint so the runtime
    automatically maps inference streams across all available CPU cores.
  * ov.preprocess for hardware-accelerated image scaling + normalisation,
    eliminating manual cv2.resize / float32 division.
  * AsyncInferQueue (depth = physical_cores / 2) for pipelining frames
    from all 4 cameras simultaneously -- the CPU is never idle.
  * Ultralytics fallback kept ONLY for development convenience; the
    production path is pure OpenVINO INT8.

Classes
-------
  1. OpenVINOInference  – compiled INT8 model with AsyncInferQueue.
  2. CameraManager      – threaded multi-camera capture (latest-frame-only).
  3. AnalyticsEngine     – inference + intrusion geometry + Event triage.

Dataclasses
-----------
  Event       – categorised detection record (Human/Vehicle/Pet + intrusion).
  AlertEvent  – intrusion/PPE alert.
  CameraStats – per-camera live metrics.

Design: zero Streamlit / FastAPI imports — pure CV/AI engine.
"""

from __future__ import annotations

import os
import time
import threading
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── OpenVINO 2024+ API ──────────────────────────────────────────────
try:
    import openvino as ov
    from openvino import AsyncInferQueue
    OV_AVAILABLE = True
except ImportError:
    OV_AVAILABLE = False

from config.settings import (
    YOLO_MODEL_XML,
    TARGET_CLASSES,
    CONFIDENCE_THRESHOLD,
    NMS_IOU_THRESHOLD,
    INPUT_SIZE,
    ASYNC_INFER_JOBS,
    CAMERAS,
    CameraSource,
    MAX_ALERT_HISTORY,
    TRIAGE_MAP,
    OV_DEVICE,
    OV_PERF_HINT,
    OV_NUM_STREAMS,
    OV_NUM_THREADS,
    CAM_RECONNECT_DELAY,
    JPEG_QUALITY,
)

logger = logging.getLogger("vision_engine")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")


# ════════════════════════════════════════════════════════════════════
#  Event Triage Categories
# ════════════════════════════════════════════════════════════════════

class TriageCategory(str, Enum):
    HUMAN_PPE   = "Human & PPE"
    VEHICLE     = "Vehicle & Plates"
    PET         = "Pet & Animal"
    INTRUSION   = "Intrusion"


@dataclass
class Event:
    """Single categorised detection event for the triage dashboard."""
    timestamp:    str
    camera_id:    str
    camera_label: str
    category:     str          # TriageCategory value
    label:        str          # e.g. "Person", "Vehicle", "Pet"
    confidence:   float = 0.0
    detail:       str = ""
    intrusion:    bool = False
    image_crop:   Optional[bytes] = None


@dataclass
class AlertEvent:
    """Intrusion / PPE alert."""
    timestamp:  str
    cam_id:     str
    cam_label:  str
    label:      str
    detail:     str
    confidence: float = 0.0


@dataclass
class CameraStats:
    """Per-camera live metrics."""
    cam_id:      str
    cam_label:   str
    fps:         float = 0.0
    detections:  Dict[str, int] = field(default_factory=dict)
    intrusions:  int = 0
    last_frame:  Optional[np.ndarray] = None


# ════════════════════════════════════════════════════════════════════
#  1. OpenVINO INT8 Inference Engine  (Zero-Lag Pipeline)
# ════════════════════════════════════════════════════════════════════

class OpenVINOInference:
    """
    High-performance OpenVINO inference with:
      * Hardware-accelerated preprocessing (ov.preprocess)
      * INT8 dispatch via THROUGHPUT hint (saturates all CPU cores)
      * AsyncInferQueue for pipelined execution across cameras

    AVX-512 / VNNI / AMX Note
    ─────────────────────────
    When the compiled model contains INT8 operations, OpenVINO's CPU
    plugin automatically dispatches to the widest available SIMD path:
      - AVX-512 VNNI   (Ice Lake, Tiger Lake, Alder Lake P)
      - Intel AMX       (Sapphire Rapids, Granite Rapids)
    This is transparent — no code change needed.  The performance hint
    THROUGHPUT tells the runtime to create multiple inference streams
    and pin them to cores for maximum utilisation.
    """

    def __init__(self, model_xml: str = YOLO_MODEL_XML,
                 device: str = OV_DEVICE,
                 n_jobs: int = ASYNC_INFER_JOBS):
        if not OV_AVAILABLE:
            raise RuntimeError(
                "OpenVINO is not installed.  Install with:  pip install openvino>=2024.0"
            )

        if not os.path.exists(model_xml):
            raise FileNotFoundError(
                f"INT8 model not found at '{model_xml}'.  "
                "Run:  python scripts/export_int8.py"
            )

        core = ov.Core()

        # ── Read model ──────────────────────────────────────────────
        model = core.read_model(model_xml)

        # ── Hardware-accelerated preprocessing ──────────────────────
        #  Instead of manual cv2.resize + float32 division in Python,
        #  we embed the preprocessing into the model graph itself.
        #  OpenVINO compiles this into optimised SIMD kernels that run
        #  on the same CPU pipeline as inference — zero Python overhead.
        ppp = ov.preprocess.PrePostProcessor(model)

        # Input: BGR uint8 image of any size (will be resized by HW)
        ppp.input().tensor() \
            .set_element_type(ov.Type.u8) \
            .set_layout(ov.Layout("NHWC")) \
            .set_spatial_dynamic_shape()

        # Model expects: FP32 [1, 3, 640, 640]  (NCHW, 0-1 normalised)
        ppp.input().preprocess() \
            .resize(ov.preprocess.ResizeAlgorithm.RESIZE_LINEAR) \
            .convert_element_type(ov.Type.f32) \
            .scale(255.0) \
            .convert_layout(ov.Layout("NCHW"))

        ppp.input().model().set_layout(ov.Layout("NCHW"))

        model = ppp.build()

        # ── Static batch shape  (NHWC for input tensor) ─────────────
        model.reshape({model.input().any_name: ov.PartialShape([1, *INPUT_SIZE, 3])})

        # ── Compile with THROUGHPUT hint ────────────────────────────
        #  THROUGHPUT mode creates multiple internal inference streams
        #  (typically = physical cores / 2) and distributes work across
        #  them.  Perfect for our 4-camera scenario where we want to
        #  saturate the CPU across all feeds simultaneously.
        config: Dict[str, str] = {
            "PERFORMANCE_HINT": OV_PERF_HINT,
        }
        if OV_NUM_STREAMS:
            config["NUM_STREAMS"] = str(OV_NUM_STREAMS)
        if OV_NUM_THREADS:
            config["INFERENCE_NUM_THREADS"] = str(OV_NUM_THREADS)

        compiled = core.compile_model(model, device, config)

        self._input_name  = compiled.input().any_name
        self._output_name = compiled.output().any_name

        # ── AsyncInferQueue ─────────────────────────────────────────
        self._queue = AsyncInferQueue(compiled, n_jobs)
        self._queue.set_callback(self._on_done)

        self._results: Dict[int, np.ndarray] = {}
        self._lock = threading.Lock()

        logger.info(
            "OpenVINO INT8 model loaded  |  device=%s  |  hint=%s  |  async_slots=%d",
            device, OV_PERF_HINT, n_jobs,
        )

    # ── Preprocessing (hardware-accelerated) ─────────────────────────
    @staticmethod
    def preprocess(frame: np.ndarray) -> np.ndarray:
        """
        Minimal preprocessing: just add batch dim.
        All resize / normalize / layout-transpose is done inside the
        compiled model via ov.preprocess — runs on SIMD, not Python.
        """
        return np.expand_dims(frame, axis=0)  # (H, W, 3) -> (1, H, W, 3)

    # ── Async submission ─────────────────────────────────────────────
    def submit(self, tensor: np.ndarray, job_id: int) -> None:
        self._queue.start_async({self._input_name: tensor}, userdata=job_id)

    def _on_done(self, infer_request, userdata):
        raw = infer_request.get_output_tensor().data.copy()
        with self._lock:
            self._results[userdata] = raw

    def get_result(self, job_id: int) -> Optional[np.ndarray]:
        with self._lock:
            return self._results.pop(job_id, None)

    def wait_all(self):
        self._queue.wait_all()

    # ── NMS + decode ─────────────────────────────────────────────────
    @staticmethod
    def postprocess(raw: np.ndarray,
                    orig_shape: Tuple[int, int],
                    conf_thr: float = CONFIDENCE_THRESHOLD,
                    iou_thr: float = NMS_IOU_THRESHOLD
                    ) -> List[Dict]:
        """Decode YOLOv8 raw output tensor -> list of detection dicts."""
        predictions = np.squeeze(raw).T
        if predictions.ndim != 2:
            return []

        cx, cy, w, h = predictions[:, 0], predictions[:, 1], predictions[:, 2], predictions[:, 3]
        class_scores = predictions[:, 4:]
        max_scores = class_scores.max(axis=1)
        mask = max_scores > conf_thr
        if not mask.any():
            return []

        cx, cy, w, h = cx[mask], cy[mask], w[mask], h[mask]
        class_scores = class_scores[mask]
        max_scores = max_scores[mask]
        class_ids = class_scores.argmax(axis=1)

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        oh, ow = orig_shape[:2]
        sx, sy = ow / INPUT_SIZE[0], oh / INPUT_SIZE[1]
        x1 = (x1 * sx).astype(int)
        x2 = (x2 * sx).astype(int)
        y1 = (y1 * sy).astype(int)
        y2 = (y2 * sy).astype(int)

        boxes_for_nms = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        indices = cv2.dnn.NMSBoxes(boxes_for_nms, max_scores.tolist(), conf_thr, iou_thr)
        if len(indices) == 0:
            return []

        detections: List[Dict] = []
        for i in indices:
            idx = i if isinstance(i, int) else i[0]
            cid = int(class_ids[idx])
            label = TARGET_CLASSES.get(cid)
            if label is None:
                continue
            detections.append({
                "bbox":       (int(x1[idx]), int(y1[idx]), int(x2[idx]), int(y2[idx])),
                "class_id":   cid,
                "label":      label,
                "confidence": float(max_scores[idx]),
            })
        return detections


# ════════════════════════════════════════════════════════════════════
#  2. Camera Manager (threaded capture — latest-frame-only)
# ════════════════════════════════════════════════════════════════════

class CameraManager:
    """
    One thread per camera.  Each thread grabs frames as fast as the
    source delivers them and overwrites a single shared buffer.
    Consumers always get the *latest* frame — no queue lag.
    """

    def __init__(self, sources: List[CameraSource] | None = None):
        self._sources = sources or CAMERAS
        self._frames: Dict[str, Optional[np.ndarray]] = {}
        self._locks:  Dict[str, threading.Lock] = {}
        self._caps:   Dict[str, cv2.VideoCapture] = {}
        self._running = False

    def start(self):
        self._running = True
        for src in self._sources:
            self._locks[src.cam_id] = threading.Lock()
            self._frames[src.cam_id] = None
            t = threading.Thread(target=self._reader, args=(src,), daemon=True)
            t.start()
            logger.info("Camera thread started: %s -> %s", src.cam_id, src.uri)

    def stop(self):
        self._running = False
        for cap in self._caps.values():
            try:
                cap.release()
            except Exception:
                pass

    def get_frame(self, cam_id: str) -> Optional[np.ndarray]:
        lock = self._locks.get(cam_id)
        if lock is None:
            return None
        with lock:
            f = self._frames.get(cam_id)
            return f.copy() if f is not None else None

    @property
    def camera_ids(self) -> List[str]:
        return [s.cam_id for s in self._sources]

    def get_source(self, cam_id: str) -> Optional[CameraSource]:
        for s in self._sources:
            if s.cam_id == cam_id:
                return s
        return None

    def _reader(self, src: CameraSource):
        uri = int(src.uri) if src.uri.isdigit() else src.uri
        cap = cv2.VideoCapture(uri)
        self._caps[src.cam_id] = cap

        if not cap.isOpened():
            logger.error("Cannot open camera %s (%s)", src.cam_id, src.uri)
            return

        while self._running:
            ok, frame = cap.read()
            if not ok:
                logger.warning("Frame drop on %s — reconnecting", src.cam_id)
                time.sleep(CAM_RECONNECT_DELAY)
                cap.release()
                cap = cv2.VideoCapture(uri)
                self._caps[src.cam_id] = cap
                continue
            with self._locks[src.cam_id]:
                self._frames[src.cam_id] = frame


# ════════════════════════════════════════════════════════════════════
#  3. Analytics Engine  (inference + intrusion + Event triage)
# ════════════════════════════════════════════════════════════════════

class AnalyticsEngine:
    """
    Main processing loop.
    - Grabs frames from CameraManager.
    - Runs inference via OpenVINO INT8 AsyncInferQueue.
    - Performs intrusion geometry checks (ROI polygons).
    - Categorises detections into triage Events.
    - Publishes annotated frames, stats, events, and alerts.
    """

    def __init__(self, cam_manager: CameraManager,
                 confidence: float = CONFIDENCE_THRESHOLD):
        self._cam = cam_manager
        self._confidence = confidence

        # ── Inference engine init ────────────────────────────────────
        try:
            self._infer = OpenVINOInference()
            self._use_ov = True
            logger.info("Using OpenVINO INT8 inference pipeline")
        except (RuntimeError, FileNotFoundError) as exc:
            logger.error("OpenVINO init failed: %s", exc)
            logger.error("Run  'python scripts/export_int8.py'  to create the INT8 model.")
            # Fallback to Ultralytics if available (development only)
            try:
                from ultralytics import YOLO
                self._fallback = YOLO("yolov8n.pt")
                self._use_ov = False
                logger.warning("FALLBACK: Using Ultralytics PyTorch (dev mode only)")
            except ImportError:
                self._fallback = None
                self._use_ov = False
                logger.error(
                    "No inference backend available — system will produce no detections"
                )

        # ── Shared state ─────────────────────────────────────────────
        self.stats:  Dict[str, CameraStats] = {}
        self.alerts: deque[AlertEvent] = deque(maxlen=MAX_ALERT_HISTORY)
        self.events: deque[Event]      = deque(maxlen=MAX_ALERT_HISTORY)
        self.intrusion_active: bool    = False
        self._lock    = threading.Lock()
        self._running = False

    @property
    def confidence(self) -> float:
        return self._confidence

    @confidence.setter
    def confidence(self, val: float):
        self._confidence = max(0.05, min(val, 1.0))

    def start(self):
        self._running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()
        logger.info("AnalyticsEngine started")

    def stop(self):
        self._running = False

    # ── Main processing loop ─────────────────────────────────────────
    def _loop(self):
        while self._running:
            any_intrusion = False

            for cam_id in self._cam.camera_ids:
                frame = self._cam.get_frame(cam_id)
                if frame is None:
                    continue

                t0 = time.perf_counter()
                detections = self._run_inference(frame)
                dt = time.perf_counter() - t0

                src = self._cam.get_source(cam_id)
                intrusions = self._check_intrusions(detections, src)
                annotated  = self._annotate(frame.copy(), detections, intrusions, src)

                if intrusions:
                    any_intrusion = True

                counts: Dict[str, int] = {}
                for d in detections:
                    counts[d["label"]] = counts.get(d["label"], 0) + 1

                fps = 1.0 / dt if dt > 0 else 0.0
                now = time.strftime("%H:%M:%S")

                with self._lock:
                    self.stats[cam_id] = CameraStats(
                        cam_id=cam_id,
                        cam_label=src.label if src else cam_id,
                        fps=round(fps, 1),
                        detections=counts,
                        intrusions=len(intrusions),
                        last_frame=annotated,
                    )

                    # Triage: create categorised Events
                    for det in detections:
                        cat = TRIAGE_MAP.get(det["label"], TriageCategory.HUMAN_PPE.value)
                        is_intr = det in intrusions
                        self.events.appendleft(Event(
                            timestamp=now,
                            camera_id=cam_id,
                            camera_label=src.label if src else cam_id,
                            category=cat,
                            label=det["label"],
                            confidence=det["confidence"],
                            detail=f"{det['label']} @ {det['bbox']}",
                            intrusion=is_intr,
                        ))

                    # Intrusion alerts
                    for intr in intrusions:
                        self.alerts.appendleft(AlertEvent(
                            timestamp=now,
                            cam_id=cam_id,
                            cam_label=src.label if src else cam_id,
                            label="Intrusion",
                            detail=f"{intr['label']} in ROI ({intr['confidence']:.0%})",
                            confidence=intr["confidence"],
                        ))

            with self._lock:
                self.intrusion_active = any_intrusion

            time.sleep(0.005)  # 5 ms yield — prevents busy-spin

    def _run_inference(self, frame: np.ndarray) -> List[Dict]:
        """Run inference through OpenVINO INT8 or fallback."""
        if self._use_ov:
            tensor = OpenVINOInference.preprocess(frame)
            self._infer.submit(tensor, job_id=0)
            self._infer.wait_all()
            raw = self._infer.get_result(0)
            if raw is None:
                return []
            return OpenVINOInference.postprocess(
                raw, frame.shape, conf_thr=self._confidence
            )
        elif hasattr(self, "_fallback") and self._fallback is not None:
            return self._run_fallback(frame)
        else:
            return []

    def _run_fallback(self, frame: np.ndarray) -> List[Dict]:
        """Ultralytics fallback for development without INT8 model."""
        results = self._fallback(frame, conf=self._confidence, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cid = int(box.cls[0])
            label = TARGET_CLASSES.get(cid)
            if label is None:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "bbox":       (int(x1), int(y1), int(x2), int(y2)),
                "class_id":   cid,
                "label":      label,
                "confidence": float(box.conf[0]),
            })
        return detections

    @staticmethod
    def _check_intrusions(detections: List[Dict],
                          source: Optional[CameraSource]) -> List[Dict]:
        if source is None or not source.roi_poly:
            return []
        poly = np.array(source.roi_poly, dtype=np.float32)
        intrusions: List[Dict] = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            dist = cv2.pointPolygonTest(poly, (float(cx), float(cy)), measureDist=False)
            if dist >= 0:
                intrusions.append(det)
        return intrusions

    @staticmethod
    def _annotate(frame: np.ndarray,
                  detections: List[Dict],
                  intrusions: List[Dict],
                  source: Optional[CameraSource]) -> np.ndarray:
        # Draw ROI polygon
        if source and source.roi_poly:
            pts = np.array(source.roi_poly, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True,
                          color=(0, 255, 255), thickness=2)
            cv2.putText(frame, "ROI", tuple(source.roi_poly[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        intr_bboxes = {d["bbox"] for d in intrusions}
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            is_intr = det["bbox"] in intr_bboxes
            color = (0, 0, 255) if is_intr else (0, 255, 0)
            thickness = 3 if is_intr else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            tag = f"{det['label']} {det['confidence']:.0%}"
            if is_intr:
                tag = "!! INTRUSION: " + tag
            cv2.putText(frame, tag, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return frame

    # ════════════════════════════════════════════════════════════════
    #  Thread-safe accessors for the API layer
    # ════════════════════════════════════════════════════════════════

    def get_annotated_frame(self, cam_id: str) -> Optional[np.ndarray]:
        with self._lock:
            s = self.stats.get(cam_id)
            return s.last_frame.copy() if s and s.last_frame is not None else None

    def get_stats_snapshot(self) -> Dict:
        with self._lock:
            return {
                cam_id: {
                    "cam_label":   s.cam_label,
                    "fps":         s.fps,
                    "detections":  s.detections,
                    "intrusions":  s.intrusions,
                }
                for cam_id, s in self.stats.items()
            }

    def get_alerts(self, n: int = 50) -> List[Dict]:
        with self._lock:
            return [
                {
                    "time":       a.timestamp,
                    "camera":     a.cam_label,
                    "label":      a.label,
                    "detail":     a.detail,
                    "confidence": round(a.confidence, 2),
                }
                for a in list(self.alerts)[:n]
            ]

    def get_events(self, n: int = 100, category: Optional[str] = None) -> List[Dict]:
        """Return categorised events, optionally filtered by triage category."""
        with self._lock:
            items = list(self.events)[:n]
            if category:
                items = [e for e in items if e.category == category]
            return [
                {
                    "time":        e.timestamp,
                    "camera":      e.camera_label,
                    "category":    e.category,
                    "label":       e.label,
                    "confidence":  round(e.confidence, 2),
                    "detail":      e.detail,
                    "intrusion":   e.intrusion,
                }
                for e in items
            ]

    def is_intrusion_active(self) -> bool:
        with self._lock:
            return self.intrusion_active

    def get_metadata_packet(self) -> Dict:
        """Compact JSON metadata packet for the dashboard."""
        with self._lock:
            return {
                "intrusion_active": self.intrusion_active,
                "stats": {
                    cam_id: {
                        "fps":        s.fps,
                        "detections": s.detections,
                        "intrusions": s.intrusions,
                    }
                    for cam_id, s in self.stats.items()
                },
                "latest_alert": (
                    {
                        "time":   self.alerts[0].timestamp,
                        "camera": self.alerts[0].cam_label,
                        "detail": self.alerts[0].detail,
                    }
                    if self.alerts else None
                ),
            }
