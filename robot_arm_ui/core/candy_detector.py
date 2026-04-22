"""Thin YOLO wrapper for candy detection.

This module is intentionally free of Qt dependencies so it can be
tested and used independently of the GUI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    from ultralytics import YOLO  # type: ignore

    _YOLO_AVAILABLE = True
except ImportError:
    YOLO = None  # type: ignore
    _YOLO_AVAILABLE = False


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


class CandyDetector:
    """Lazy-loaded YOLO detector.

    Usage::

        det = CandyDetector()
        det.load("path/to/best.pt")
        results = det.detect(bgr_frame, conf=0.5)
    """

    def __init__(self) -> None:
        self._model: Optional[object] = None
        self.class_names: list[str] = []
        self.model_path: str = ""

    # ------------------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @staticmethod
    def is_available() -> bool:
        """Return True if the ultralytics package is importable."""
        return _YOLO_AVAILABLE

    # ------------------------------------------------------------------
    def load(self, path: str) -> None:
        """Load a YOLO model from *path*.

        Raises ``RuntimeError`` if ultralytics is not installed, or if
        the model file cannot be loaded.
        """
        if not _YOLO_AVAILABLE:
            raise RuntimeError("ultralytics is not installed — run: pip install ultralytics")

        model = YOLO(path)
        # Read class names from the model
        self.class_names = list(model.names.values()) if hasattr(model, "names") else []
        self._model = model
        self.model_path = path

    def unload(self) -> None:
        self._model = None
        self.class_names = []
        self.model_path = ""

    # ------------------------------------------------------------------
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
            The input image in BGR colour order.
        conf : float
            Minimum confidence threshold.
        class_filter : int or None
            If set, only return detections of this class ID.

        Returns
        -------
        list[Detection]
            Detections in image-pixel coordinates.
        """
        if self._model is None:
            return []

        results = self._model(bgr_frame, conf=conf, verbose=False)[0]

        detections: list[Detection] = []
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
                    class_name=self.class_names[cls] if cls < len(self.class_names) else str(cls),
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
