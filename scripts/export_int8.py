#!/usr/bin/env python3
"""
scripts/export_int8.py
──────────────────────
Post-Training Quantization (PTQ) script for YOLOv8n → OpenVINO INT8.

WHY INT8?
─────────
  • INT8 reduces model size by ~4× vs FP32 (6 MB vs 24 MB on disk).
  • INT8 arithmetic fits inside CPU L1/L2 caches far more efficiently,
    eliminating memory-bandwidth bottlenecks on edge devices.
  • Modern Intel CPUs (10th-gen+) execute INT8 via VNNI (AVX-512_VNNI)
    or AMX instructions, delivering 2-4× higher throughput than FP32
    for matrix-multiply-accumulate operations.
  • The result: 30-60 FPS inference on a 4-core i5 without any GPU.

USAGE
─────
    cd <project root>
    python scripts/export_int8.py

    The script will:
      1.  Load yolov8n.pt (PyTorch weights).
      2.  Export to OpenVINO IR with INT8 quantisation via Ultralytics'
          built-in NNCF integration (Post-Training Quantization).
      3.  Save the result to models/yolov8n_openvino_model/.

    After export, the backend will automatically pick up the INT8 model
    from YOLO_MODEL_XML in config/settings.py.
"""

from __future__ import annotations

import os
import sys
import shutil
from pathlib import Path

# ── Resolve project root ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# ── Configuration ────────────────────────────────────────────────────
PT_WEIGHTS      = PROJECT_ROOT / "yolov8n.pt"
OUTPUT_DIR      = PROJECT_ROOT / "models" / "yolov8n_openvino_model"
EXPORT_IMGSZ    = 640  # match INPUT_SIZE in settings.py


def main():
    # ----------------------------------------------------------------
    #  Pre-flight checks
    # ----------------------------------------------------------------
    if not PT_WEIGHTS.exists():
        print(f"[ERROR] PyTorch weights not found at: {PT_WEIGHTS}")
        print("        Download with:  yolo detect download model=yolov8n.pt")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] 'ultralytics' package not installed.  pip install ultralytics")
        sys.exit(1)

    try:
        import openvino  # noqa: F401
    except ImportError:
        print("[ERROR] 'openvino' package not installed.  pip install openvino>=2024.0")
        sys.exit(1)

    try:
        import nncf  # noqa: F401
        print(f"[INFO]  NNCF version : {nncf.__version__}")
    except ImportError:
        print("[WARN]  'nncf' not found — installing now …")
        os.system(f"{sys.executable} -m pip install nncf")

    # ----------------------------------------------------------------
    #  Step 1 — Load model
    # ----------------------------------------------------------------
    print(f"\n{'═' * 60}")
    print(f"  Loading YOLOv8n from  {PT_WEIGHTS}")
    print(f"{'═' * 60}\n")

    model = YOLO(str(PT_WEIGHTS))

    # ----------------------------------------------------------------
    #  Step 2 — Export to OpenVINO INT8
    # ----------------------------------------------------------------
    #  Ultralytics ≥ 8.1 integrates NNCF for post-training quantisation.
    #  Setting  int8=True  triggers NNCF's PTQ pipeline automatically:
    #    • A small calibration dataset (COCO128) is downloaded if needed.
    #    • Activation ranges are collected per-layer.
    #    • Weights + activations are quantised to symmetric INT8.
    #
    #  The resulting IR model uses INT8 for Conv/MatMul ops while keeping
    #  sensitive layers (first/last convolutions) in FP16/FP32 to
    #  preserve accuracy — this is NNCF's "mixed-precision" strategy.
    # ----------------------------------------------------------------

    print(f"\n{'═' * 60}")
    print("  Exporting to OpenVINO INT8 (Post-Training Quantisation)")
    print("  This may take 2-5 minutes on first run (calibration).")
    print(f"{'═' * 60}\n")

    export_path = model.export(
        format="openvino",
        int8=True,                # ← triggers NNCF PTQ
        imgsz=EXPORT_IMGSZ,
        half=False,               # INT8 supersedes FP16; keep FP32 baseline
        dynamic=False,            # static shape → better CPU optimisation
        simplify=True,            # ONNX graph simplification before IR
    )

    print(f"\n[OK]  Export complete → {export_path}")

    # ----------------------------------------------------------------
    #  Step 3 — Move to canonical location
    # ----------------------------------------------------------------
    export_path = Path(export_path)

    if export_path.resolve() != OUTPUT_DIR.resolve():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for f in export_path.glob("*"):
            dest = OUTPUT_DIR / f.name
            if dest.exists():
                dest.unlink()
            shutil.move(str(f), str(dest))
        print(f"[OK]  Model files moved to {OUTPUT_DIR}")

    # ----------------------------------------------------------------
    #  Verification
    # ----------------------------------------------------------------
    xml_file = OUTPUT_DIR / "yolov8n.xml"
    bin_file = OUTPUT_DIR / "yolov8n.bin"

    if xml_file.exists() and bin_file.exists():
        xml_mb = xml_file.stat().st_size / (1024 * 1024)
        bin_mb = bin_file.stat().st_size / (1024 * 1024)
        print(f"\n{'─' * 60}")
        print(f"  IR XML  : {xml_file}  ({xml_mb:.1f} MB)")
        print(f"  IR BIN  : {bin_file}  ({bin_mb:.1f} MB)")
        print(f"  Total   : {xml_mb + bin_mb:.1f} MB")
        print(f"{'─' * 60}")
        print(f"\n  ✅  INT8 model ready.  The backend will load it automatically.")
        print(f"      Config path: YOLO_MODEL_XML = \"{xml_file.relative_to(PROJECT_ROOT)}\"")
    else:
        print(f"\n  ⚠️  Expected files not found in {OUTPUT_DIR}.")
        print(f"      Check the export output above for errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
