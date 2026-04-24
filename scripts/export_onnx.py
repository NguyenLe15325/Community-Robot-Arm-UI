#!/usr/bin/env python3
"""Export a YOLO .pt model to ONNX format for lightweight deployment.

This script converts an ultralytics YOLO model (.pt) into ONNX format
(.onnx) so it can be loaded by the Community Robot Arm UI without
requiring the full ultralytics/PyTorch stack.

Requirements (for export only — not needed for the final app):
    pip install ultralytics

Usage:
    python scripts/export_onnx.py path/to/best.pt
    python scripts/export_onnx.py path/to/best.pt --imgsz 640
    python scripts/export_onnx.py path/to/best.pt --output my_model.onnx

The exported .onnx file can then be loaded in the app via:
    Vision Tab → Load Model → select the .onnx file

Supported model types:
    - Object Detection  (yolov8n.pt, best.pt, etc.)
    - Instance Segmentation (yolov8n-seg.pt) — bounding boxes extracted
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a YOLO .pt model to ONNX for lightweight deployment."
    )
    parser.add_argument(
        "model",
        help="Path to the YOLO .pt model file.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size for the model (default: 640).",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="",
        help="Output .onnx file path (default: same name as input).",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Simplify the ONNX graph (requires onnxsim).",
    )
    args = parser.parse_args()

    # Check ultralytics is available
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics is not installed.")
        print("Install with:  pip install ultralytics")
        return 1

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}")
        return 1

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    # Show model info
    if hasattr(model, "names"):
        names = model.names
        print(f"Classes ({len(names)}): {list(names.values())}")

    # Export to ONNX
    print(f"Exporting to ONNX (imgsz={args.imgsz})...")
    export_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        simplify=args.simplify,
    )

    # Rename if custom output specified
    if args.output:
        output_path = Path(args.output)
        Path(export_path).rename(output_path)
        export_path = str(output_path)

    print(f"\nExport complete: {export_path}")
    print(f"File size: {Path(export_path).stat().st_size / 1024 / 1024:.1f} MB")
    print(f"\nLoad this file in the app:")
    print(f"    Vision Tab → Load Model → {Path(export_path).name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
