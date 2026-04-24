"""Object detector with dual-backend support: ONNX Runtime and ultralytics.

ONNX Runtime (recommended for deployment):
    - Lightweight (~30 MB), no PyTorch dependency
    - Load .onnx models exported from ultralytics
    - Always available in bundled executables

ultralytics (development / training):
    - Full YOLO framework with .pt model support
    - Optional; only used if installed and a .pt file is loaded

This module is free of Qt dependencies so it can be tested independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# ── Optional backend imports ─────────────────────────────────────────
try:
    import onnxruntime as ort  # type: ignore

    _ORT_AVAILABLE = True
except ImportError:
    ort = None  # type: ignore
    _ORT_AVAILABLE = False

try:
    from ultralytics import YOLO  # type: ignore

    _YOLO_AVAILABLE = True
except ImportError:
    YOLO = None  # type: ignore
    _YOLO_AVAILABLE = False


# ── Data classes ─────────────────────────────────────────────────────

@dataclass
class Detection:
    """A single detection result in image-pixel coordinates."""

    class_id: int
    class_name: str
    confidence: float
    cx: float  # centroid X (pixels)
    cy: float  # centroid Y (pixels)
    x1: int
    y1: int
    x2: int
    y2: int


# ── CandyDetector ────────────────────────────────────────────────────

class CandyDetector:
    """Lazy-loaded object detector with ONNX + ultralytics backends.

    Priority order when loading:
        1. ``.onnx`` → uses ONNX Runtime (lightweight, always bundled)
        2. ``.pt``   → uses ultralytics (requires ``pip install ultralytics``)

    Usage::

        det = CandyDetector()
        det.load("best.onnx")          # or "best.pt"
        results = det.detect(bgr_frame, conf=0.5)
    """

    def __init__(self) -> None:
        self._backend: Optional[str] = None     # "onnx" or "yolo"
        self._ort_session: Optional[object] = None
        self._yolo_model: Optional[object] = None
        self._input_size: tuple[int, int] = (640, 640)
        self._end2end: bool = False
        self.class_names: list[str] = []
        self.model_path: str = ""

    # ── Status ───────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._backend is not None

    @staticmethod
    def is_available() -> bool:
        """Return True if at least one backend is importable."""
        return _ORT_AVAILABLE or _YOLO_AVAILABLE

    @staticmethod
    def onnx_available() -> bool:
        return _ORT_AVAILABLE

    @staticmethod
    def yolo_available() -> bool:
        return _YOLO_AVAILABLE

    @staticmethod
    def supported_formats() -> str:
        """File dialog filter string listing supported formats."""
        parts = []
        if _ORT_AVAILABLE:
            parts.append("ONNX Models (*.onnx)")
        if _YOLO_AVAILABLE:
            parts.append("PyTorch Weights (*.pt)")
        parts.append("All Files (*.*)")
        return ";;".join(parts)

    # ── Load / unload ────────────────────────────────────────────────

    def load(self, path: str) -> None:
        """Load a model from *path*.

        Dispatches to the correct backend based on file extension:
            - ``.onnx`` → ONNX Runtime
            - ``.pt``   → ultralytics YOLO

        Raises ``RuntimeError`` if the required backend is missing.
        """
        ext = Path(path).suffix.lower()

        if ext == ".onnx":
            self._load_onnx(path)
        elif ext == ".pt":
            self._load_yolo(path)
        else:
            # Try ONNX first, then YOLO
            if _ORT_AVAILABLE:
                self._load_onnx(path)
            elif _YOLO_AVAILABLE:
                self._load_yolo(path)
            else:
                raise RuntimeError(
                    "No inference backend available.\n"
                    "Install onnxruntime (pip install onnxruntime) or "
                    "ultralytics (pip install ultralytics)."
                )

    def _load_onnx(self, path: str) -> None:
        """Load an ONNX model via ONNX Runtime."""
        if not _ORT_AVAILABLE:
            raise RuntimeError(
                "onnxruntime is not installed.\n"
                "Install with: pip install onnxruntime"
            )

        session = ort.InferenceSession(
            path, providers=["CPUExecutionProvider"]
        )

        # Read metadata (class names, input size) from the ONNX model.
        # ultralytics embeds metadata when exporting with `yolo export`.
        meta = session.get_modelmeta().custom_metadata_map

        # Class names: stored as a Python dict string in metadata
        names_str = meta.get("names", "")
        if names_str:
            # Format: "{0: 'class0', 1: 'class1', ...}"
            import ast
            try:
                names_dict = ast.literal_eval(names_str)
                self.class_names = [
                    names_dict[k] for k in sorted(names_dict.keys())
                ]
            except Exception:
                self.class_names = []
        else:
            self.class_names = []

        # Input size from model
        input_shape = session.get_inputs()[0].shape  # e.g. [1, 3, 640, 640]
        if len(input_shape) == 4:
            self._input_size = (int(input_shape[2]), int(input_shape[3]))

        # Detect end2end format: NMS baked into model, output is [1, N, 6]
        self._end2end = meta.get("end2end", "").lower() in ("true", "1")

        self._ort_session = session
        self._yolo_model = None
        self._backend = "onnx"
        self.model_path = path

    def _load_yolo(self, path: str) -> None:
        """Load a YOLO .pt model via ultralytics."""
        if not _YOLO_AVAILABLE:
            raise RuntimeError(
                "ultralytics is not installed.\n"
                "Install with: pip install ultralytics\n\n"
                "Alternatively, export your model to ONNX format:\n"
                "    yolo export model=best.pt format=onnx\n"
                "Then load the .onnx file instead."
            )

        model = YOLO(path)
        self.class_names = (
            list(model.names.values()) if hasattr(model, "names") else []
        )
        self._yolo_model = model
        self._ort_session = None
        self._backend = "yolo"
        self.model_path = path

    def unload(self) -> None:
        self._ort_session = None
        self._yolo_model = None
        self._backend = None
        self.class_names = []
        self.model_path = ""

    # ── Inference ────────────────────────────────────────────────────

    def detect(
        self,
        bgr_frame,
        conf: float = 0.5,
        class_filter: Optional[int] = None,
    ) -> list[Detection]:
        """Run inference on a BGR image.

        Parameters
        ----------
        bgr_frame : np.ndarray
            The input image in BGR colour order (H, W, 3).
        conf : float
            Minimum confidence threshold.
        class_filter : int or None
            If set, only return detections of this class ID.

        Returns
        -------
        list[Detection]
            Detections in image-pixel coordinates.
        """
        if self._backend == "onnx":
            return self._detect_onnx(bgr_frame, conf, class_filter)
        elif self._backend == "yolo":
            return self._detect_yolo(bgr_frame, conf, class_filter)
        return []

    # ── ONNX inference ───────────────────────────────────────────────

    def _detect_onnx(
        self, bgr_frame, conf: float, class_filter: Optional[int]
    ) -> list[Detection]:
        """Run ONNX Runtime inference.

        Supports two output formats:
          - **End2end** (end2end=True): shape [1, N, 6]
            Each row: [x1, y1, x2, y2, score, class_id]
            NMS already applied inside the model.
          - **Standard**: shape [1, 4+n_cls, N]
            Rows 0-3: cx, cy, w, h; rows 4+: class scores.
            Requires external NMS.
        """
        import cv2

        h_orig, w_orig = bgr_frame.shape[:2]
        ih, iw = self._input_size

        # Preprocess: resize, BGR->RGB, HWC->CHW, normalize 0-1, add batch dim
        resized = cv2.resize(bgr_frame, (iw, ih))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, H, W)

        # Run inference
        input_name = self._ort_session.get_inputs()[0].name
        outputs = self._ort_session.run(None, {input_name: blob})
        output = outputs[0]

        # Scale factors from input size back to original image
        sx = w_orig / iw
        sy = h_orig / ih

        # Detect output format
        if self._end2end:
            return self._parse_end2end(output, conf, class_filter, sx, sy)
        else:
            return self._parse_standard(output, conf, class_filter, sx, sy)

    def _parse_end2end(
        self, output, conf, class_filter, sx, sy
    ) -> list[Detection]:
        """Parse end2end ONNX output: [1, N, 6] = [x1, y1, x2, y2, score, class_id]."""
        preds = output[0]  # (N, 6)
        detections: list[Detection] = []

        for row in preds:
            x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            score = float(row[4])
            cls_id = int(row[5])

            if score < conf:
                continue
            if class_filter is not None and cls_id != class_filter:
                continue

            # Scale from input coords to original image coords
            x1i = int(x1 * sx)
            y1i = int(y1 * sy)
            x2i = int(x2 * sx)
            y2i = int(y2 * sy)

            detections.append(
                Detection(
                    class_id=cls_id,
                    class_name=(
                        self.class_names[cls_id]
                        if cls_id < len(self.class_names)
                        else str(cls_id)
                    ),
                    confidence=score,
                    cx=(x1i + x2i) / 2.0,
                    cy=(y1i + y2i) / 2.0,
                    x1=x1i,
                    y1=y1i,
                    x2=x2i,
                    y2=y2i,
                )
            )

        return detections

    def _parse_standard(
        self, output, conf, class_filter, sx, sy
    ) -> list[Detection]:
        """Parse standard ONNX output: [1, 4+n_cls, N] = cx,cy,w,h + class scores."""
        import cv2

        preds = output[0].T  # (N, 4+n_cls)
        n_cls = len(self.class_names) if self.class_names else (preds.shape[1] - 4)

        boxes, scores, class_ids = [], [], []

        for row in preds:
            cx, cy, w, h = row[:4]
            class_scores = row[4:4 + n_cls]
            cls_id = int(np.argmax(class_scores))
            score = float(class_scores[cls_id])

            if score < conf:
                continue
            if class_filter is not None and cls_id != class_filter:
                continue

            x1 = int((cx - w / 2) * sx)
            y1 = int((cy - h / 2) * sy)
            x2 = int((cx + w / 2) * sx)
            y2 = int((cy + h / 2) * sy)

            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(score)
            class_ids.append(cls_id)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, scores, conf, 0.5)

        detections: list[Detection] = []
        for idx in indices:
            i = int(idx)
            x, y, bw, bh = boxes[i]
            detections.append(
                Detection(
                    class_id=class_ids[i],
                    class_name=(
                        self.class_names[class_ids[i]]
                        if class_ids[i] < len(self.class_names)
                        else str(class_ids[i])
                    ),
                    confidence=scores[i],
                    cx=x + bw / 2.0,
                    cy=y + bh / 2.0,
                    x1=x,
                    y1=y,
                    x2=x + bw,
                    y2=y + bh,
                )
            )

        return detections

    # ── YOLO (ultralytics) inference ─────────────────────────────────

    def _detect_yolo(
        self, bgr_frame, conf: float, class_filter: Optional[int]
    ) -> list[Detection]:
        """Run ultralytics YOLO inference (original path)."""
        results = self._yolo_model(bgr_frame, conf=conf, verbose=False)[0]

        detections: list[Detection] = []
        if results.boxes is None:
            return detections
        for box in results.boxes:
            cls = int(box.cls[0])
            if class_filter is not None and cls != class_filter:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            detections.append(
                Detection(
                    class_id=cls,
                    class_name=(
                        self.class_names[cls]
                        if cls < len(self.class_names)
                        else str(cls)
                    ),
                    confidence=float(box.conf[0]),
                    cx=cx,
                    cy=cy,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

        return detections
