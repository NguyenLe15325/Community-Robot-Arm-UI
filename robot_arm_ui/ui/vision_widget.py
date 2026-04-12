from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from PyQt6.QtCore import QPoint, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QMouseEvent, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

try:
    import cv2  # type: ignore
    import numpy as np

    CV_AVAILABLE = True
except Exception:
    cv2 = None
    np = None
    CV_AVAILABLE = False


@dataclass
class CornerWorld:
    x: float
    y: float


class ImageLabel(QLabel):
    mouse_moved = pyqtSignal(int, int)
    mouse_clicked = pyqtSignal(int, int)

    def __init__(self, title: str) -> None:
        super().__init__(title)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)
        self.setMinimumSize(620, 420)
        self.setObjectName("VisionFrame")

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        self.mouse_moved.emit(event.position().toPoint().x(), event.position().toPoint().y())
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self.mouse_clicked.emit(event.position().toPoint().x(), event.position().toPoint().y())
        super().mousePressEvent(event)


class VisionWidget(QWidget):
    move_request_world_xy = pyqtSignal(float, float, float, float)

    def __init__(self) -> None:
        super().__init__()

        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._update_frame)

        self._cap = None
        self._latest_roi_size: tuple[int, int] | None = None
        self._latest_roi_to_world = None
        self._cursor_world: tuple[float, float] | None = None
        self._stable_ordered_corners = None
        self._tracking_gray = None
        self._tracking_miss_count = 0
        self._last_roi_frame = None
        self._roi_hold_frames = 8
        self._corner_smoothing = 0.25

        self._build_ui()
        self._connect_signals()

        if not CV_AVAILABLE:
            self.status_label.setText("OpenCV not available. Install opencv-contrib-python and numpy.")
            self.start_button.setEnabled(False)

    def stop(self) -> None:
        self._timer.stop()
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._latest_roi_size = None
        self._latest_roi_to_world = None
        self._stable_ordered_corners = None
        self._tracking_gray = None
        self._tracking_miss_count = 0
        self._last_roi_frame = None

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(12)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(8, 8, 8, 8)
        content_layout.setSpacing(12)

        content_layout.addWidget(self._build_controls_group())

        frame_row = QHBoxLayout()
        self.original_frame_label = ImageLabel("Original frame")
        self.roi_frame_label = ImageLabel("ROI frame (click to sample XY)")
        frame_row.addWidget(self.original_frame_label, stretch=1)
        frame_row.addWidget(self.roi_frame_label, stretch=1)
        content_layout.addLayout(frame_row)

        content_layout.addWidget(self._build_cursor_group())
        content_layout.addStretch(1)

        root.addWidget(self._wrap_scroll(content))

    def _wrap_scroll(self, content: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QScrollArea.Shape.NoFrame)
        area.setWidget(content)
        return area

    def _build_controls_group(self) -> QGroupBox:
        group = QGroupBox("Vision")
        layout = QGridLayout(group)

        self.camera_index_spin = QSpinBox()
        self.camera_index_spin.setRange(0, 10)
        self.camera_index_spin.setValue(0)

        self.start_button = QPushButton("Start Camera")
        self.stop_button = QPushButton("Stop Camera")

        self.show_ids_check = QCheckBox("Show marker IDs")
        self.show_ids_check.setChecked(True)

        self.settings_toggle = QCheckBox("Show advanced ROI settings")
        self.settings_toggle.setChecked(True)

        self.roi_width_spin = QSpinBox()
        self.roi_width_spin.setRange(200, 1600)
        self.roi_width_spin.setValue(700)

        self.roi_height_spin = QSpinBox()
        self.roi_height_spin.setRange(200, 1200)
        self.roi_height_spin.setValue(500)

        # Marker IDs for corners in world axes order
        self.id_bl_spin = QSpinBox()
        self.id_br_spin = QSpinBox()
        self.id_tr_spin = QSpinBox()
        self.id_tl_spin = QSpinBox()
        for spin in (self.id_bl_spin, self.id_br_spin, self.id_tr_spin, self.id_tl_spin):
            spin.setRange(0, 49)
        self.id_bl_spin.setValue(0)
        self.id_br_spin.setValue(1)
        self.id_tr_spin.setValue(2)
        self.id_tl_spin.setValue(3)

        self.anchor_mode_combo = QComboBox()
        self.anchor_mode_combo.addItems(["Center", "Corner"])

        self.corner_bl_combo = QComboBox()
        self.corner_br_combo = QComboBox()
        self.corner_tr_combo = QComboBox()
        self.corner_tl_combo = QComboBox()
        for combo in (
            self.corner_bl_combo,
            self.corner_br_combo,
            self.corner_tr_combo,
            self.corner_tl_combo,
        ):
            combo.addItems(["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"])

        # Typical default for tags placed outside ROI: choose inward-facing marker corners.
        self.corner_bl_combo.setCurrentIndex(1)
        self.corner_br_combo.setCurrentIndex(0)
        self.corner_tr_combo.setCurrentIndex(3)
        self.corner_tl_combo.setCurrentIndex(2)

        layout.addWidget(QLabel("Camera index"), 0, 0)
        layout.addWidget(self.camera_index_spin, 0, 1)
        layout.addWidget(self.start_button, 0, 2)
        layout.addWidget(self.stop_button, 0, 3)
        layout.addWidget(self.show_ids_check, 0, 4)
        layout.addWidget(self.settings_toggle, 0, 5)

        self.lbl_roi_width = QLabel("ROI width (px)")
        self.lbl_roi_height = QLabel("ROI height (px)")
        self.lbl_id_bl = QLabel("ID bottom-left")
        self.lbl_id_br = QLabel("ID bottom-right")
        self.lbl_id_tr = QLabel("ID top-right")
        self.lbl_id_tl = QLabel("ID top-left")
        self.lbl_anchor_mode = QLabel("Anchor point mode")
        self.lbl_corner_bl = QLabel("BL corner")
        self.lbl_corner_br = QLabel("BR corner")
        self.lbl_corner_tr = QLabel("TR corner")
        self.lbl_corner_tl = QLabel("TL corner")

        layout.addWidget(self.lbl_roi_width, 1, 0)
        layout.addWidget(self.roi_width_spin, 1, 1)
        layout.addWidget(self.lbl_roi_height, 1, 2)
        layout.addWidget(self.roi_height_spin, 1, 3)

        layout.addWidget(self.lbl_id_bl, 2, 0)
        layout.addWidget(self.id_bl_spin, 2, 1)
        layout.addWidget(self.lbl_id_br, 2, 2)
        layout.addWidget(self.id_br_spin, 2, 3)
        layout.addWidget(self.lbl_id_tr, 3, 0)
        layout.addWidget(self.id_tr_spin, 3, 1)
        layout.addWidget(self.lbl_id_tl, 3, 2)
        layout.addWidget(self.id_tl_spin, 3, 3)

        layout.addWidget(self.lbl_anchor_mode, 4, 0)
        layout.addWidget(self.anchor_mode_combo, 4, 1)
        layout.addWidget(self.lbl_corner_bl, 4, 2)
        layout.addWidget(self.corner_bl_combo, 4, 3)
        layout.addWidget(self.lbl_corner_br, 5, 0)
        layout.addWidget(self.corner_br_combo, 5, 1)
        layout.addWidget(self.lbl_corner_tr, 5, 2)
        layout.addWidget(self.corner_tr_combo, 5, 3)
        layout.addWidget(self.lbl_corner_tl, 6, 0)
        layout.addWidget(self.corner_tl_combo, 6, 1)

        self.status_label = QLabel("Waiting for camera")
        layout.addWidget(self.status_label, 7, 0, 1, 5)

        return group

    def _build_cursor_group(self) -> QGroupBox:
        group = QGroupBox("ROI World Coordinates")
        layout = QHBoxLayout(group)

        world_form = QFormLayout()
        self.world_bl_x = self._float_spin(0.0)
        self.world_bl_y = self._float_spin(0.0)
        self.world_br_x = self._float_spin(300.0)
        self.world_br_y = self._float_spin(0.0)
        self.world_tr_x = self._float_spin(300.0)
        self.world_tr_y = self._float_spin(300.0)
        self.world_tl_x = self._float_spin(0.0)
        self.world_tl_y = self._float_spin(300.0)

        world_form.addRow("BL world X", self.world_bl_x)
        world_form.addRow("BL world Y", self.world_bl_y)
        world_form.addRow("BR world X", self.world_br_x)
        world_form.addRow("BR world Y", self.world_br_y)
        world_form.addRow("TR world X", self.world_tr_x)
        world_form.addRow("TR world Y", self.world_tr_y)
        world_form.addRow("TL world X", self.world_tl_x)
        world_form.addRow("TL world Y", self.world_tl_y)

        cursor_form = QFormLayout()
        self.cursor_px_label = QLabel("--")
        self.cursor_world_label = QLabel("X: --  Y: --")
        self.cursor_z_spin = self._float_spin(0.0)
        self.cursor_feed_spin = self._float_spin(30.0)
        self.send_cursor_button = QPushButton("Queue Move To Cursor XY")

        cursor_form.addRow("Cursor ROI (px)", self.cursor_px_label)
        cursor_form.addRow("Cursor World", self.cursor_world_label)
        cursor_form.addRow("Target Z (world)", self.cursor_z_spin)
        cursor_form.addRow("Feed (deg/s)", self.cursor_feed_spin)
        cursor_form.addRow(self.send_cursor_button)

        layout.addLayout(world_form)
        layout.addLayout(cursor_form)

        return group

    def _connect_signals(self) -> None:
        self.start_button.clicked.connect(self._start_camera)
        self.stop_button.clicked.connect(self.stop)
        self.roi_frame_label.mouse_clicked.connect(self._on_roi_mouse_click)
        self.send_cursor_button.clicked.connect(self._emit_move_request)
        self.settings_toggle.toggled.connect(self._set_advanced_settings_visible)

    def _set_advanced_settings_visible(self, visible: bool) -> None:
        widgets = [
            self.lbl_roi_width,
            self.lbl_roi_height,
            self.lbl_id_bl,
            self.lbl_id_br,
            self.lbl_id_tr,
            self.lbl_id_tl,
            self.lbl_anchor_mode,
            self.lbl_corner_bl,
            self.lbl_corner_br,
            self.lbl_corner_tr,
            self.lbl_corner_tl,
            self.roi_width_spin,
            self.roi_height_spin,
            self.id_bl_spin,
            self.id_br_spin,
            self.id_tr_spin,
            self.id_tl_spin,
            self.anchor_mode_combo,
            self.corner_bl_combo,
            self.corner_br_combo,
            self.corner_tr_combo,
            self.corner_tl_combo,
        ]
        for widget in widgets:
            widget.setVisible(visible)

    def _float_spin(self, value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-10000.0, 10000.0)
        spin.setDecimals(3)
        spin.setSingleStep(1.0)
        spin.setValue(value)
        spin.setMinimumWidth(120)
        return spin

    def _start_camera(self) -> None:
        if not CV_AVAILABLE:
            return

        self.stop()
        index = self.camera_index_spin.value()

        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            self.status_label.setText(f"Cannot open camera {index}")
            return

        self._cap = cap
        self.status_label.setText(f"Camera {index} running")
        self._timer.start()

    def _update_frame(self) -> None:
        if self._cap is None:
            return

        ok, frame = self._cap.read()
        if not ok:
            self.status_label.setText("Camera read failed")
            return
        overlay = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ordered = self._detect_ordered_corners(overlay)
        if ordered is not None:
            stable = self._stabilize_ordered_corners(ordered)
            self._stable_ordered_corners = stable
            self._tracking_gray = gray.copy()
            self._tracking_miss_count = 0
            self._render_roi(frame, overlay, stable, "ROI detected")
            return

        tracked = self._track_ordered_corners(gray)
        if tracked is not None:
            stable = self._stabilize_ordered_corners(tracked)
            self._stable_ordered_corners = stable
            self._tracking_gray = gray.copy()
            self._tracking_miss_count = 0
            self._render_roi(frame, overlay, stable, "ROI tracked")
            return

        if self._stable_ordered_corners is not None and self._tracking_miss_count < self._roi_hold_frames:
            self._tracking_miss_count += 1
            self._render_roi(frame, overlay, self._stable_ordered_corners, "ROI held")
            return

        self._tracking_miss_count += 1
        self._latest_roi_size = None
        self._latest_roi_to_world = None
        self._last_roi_frame = None
        self.roi_frame_label.setText("ROI frame\n(need 4 ArUco anchors visible)")
        self.status_label.setText("ROI lost")

        self._set_image(self.original_frame_label, overlay)

    def _render_roi(self, frame, overlay, ordered, status_text: str) -> None:
        polygon = ordered.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [polygon], isClosed=True, color=(0, 255, 255), thickness=2)

        roi_w = self.roi_width_spin.value()
        roi_h = self.roi_height_spin.value()
        dst = self._roi_destination_points(roi_w, roi_h)

        h_src_to_roi = cv2.getPerspectiveTransform(ordered.astype(np.float32), dst.astype(np.float32))
        roi = cv2.warpPerspective(frame, h_src_to_roi, (roi_w, roi_h))
        self._last_roi_frame = roi
        self._set_image(self.roi_frame_label, roi)
        self._latest_roi_size = (roi_w, roi_h)

        world_quad = self._world_quad()
        self._latest_roi_to_world = cv2.getPerspectiveTransform(dst.astype(np.float32), world_quad.astype(np.float32))
        self.status_label.setText(status_text)
        self._set_image(self.original_frame_label, overlay)

    def _stabilize_ordered_corners(self, ordered) -> Any:
        if self._stable_ordered_corners is None:
            return ordered.astype(np.float32)

        alpha = self._corner_smoothing
        previous = self._stable_ordered_corners.astype(np.float32)
        current = ordered.astype(np.float32)
        return previous * (1.0 - alpha) + current * alpha

    def _track_ordered_corners(self, current_gray) -> Optional[Any]:
        if self._tracking_gray is None or self._stable_ordered_corners is None or not CV_AVAILABLE:
            return None

        previous_points = self._stable_ordered_corners.astype(np.float32).reshape(-1, 1, 2)
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self._tracking_gray,
            current_gray,
            previous_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        if next_points is None or status is None:
            return None

        next_points = next_points.reshape(-1, 2)
        status = status.reshape(-1).astype(bool)
        good_prev = previous_points.reshape(-1, 2)[status]
        good_next = next_points[status]

        if len(good_next) >= 4:
            candidate = next_points
        elif len(good_next) >= 2:
            affine, _ = cv2.estimateAffinePartial2D(good_prev, good_next, method=cv2.RANSAC, ransacReprojThreshold=3.0)
            if affine is None:
                return None
            ones = np.ones((previous_points.shape[0], 1), dtype=np.float32)
            expanded = np.concatenate([previous_points.reshape(-1, 2), ones], axis=1)
            candidate = (expanded @ affine.T).astype(np.float32)
        else:
            return None

        if not self._is_valid_quad(candidate):
            return None

        return candidate

    def _is_valid_quad(self, ordered) -> bool:
        if ordered is None:
            return False

        quad = ordered.astype(np.float32)
        if np.any(~np.isfinite(quad)):
            return False

        area = abs(cv2.contourArea(quad.reshape((-1, 1, 2))))
        return area >= 100.0

    def _detect_ordered_corners(self, frame) -> Optional[Any]:
        if not CV_AVAILABLE:
            return None

        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(frame)

        if ids is None or len(ids) == 0:
            return None

        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        id_to_center: dict[int, Any] = {}
        id_to_corners: dict[int, Any] = {}
        for marker_corners, marker_id in zip(corners, ids.flatten().tolist()):
            pts = marker_corners.reshape(4, 2)
            center = pts.mean(axis=0)
            id_to_center[int(marker_id)] = center
            id_to_corners[int(marker_id)] = pts

            if self.show_ids_check.isChecked():
                c = center.astype(int)
                cv2.putText(
                    frame,
                    str(marker_id),
                    (int(c[0]) + 6, int(c[1]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.circle(frame, (int(c[0]), int(c[1])), 4, (255, 100, 0), -1)

        ordered_ids = [
            self.id_bl_spin.value(),
            self.id_br_spin.value(),
            self.id_tr_spin.value(),
            self.id_tl_spin.value(),
        ]

        corner_combos = [
            self.corner_bl_combo,
            self.corner_br_combo,
            self.corner_tr_combo,
            self.corner_tl_combo,
        ]

        if not all(marker_id in id_to_center for marker_id in ordered_ids):
            return None

        use_center = self.anchor_mode_combo.currentText().lower() == "center"
        ordered_points = []
        for marker_id, corner_combo in zip(ordered_ids, corner_combos):
            if use_center:
                point = id_to_center[marker_id]
            else:
                corner_index = corner_combo.currentIndex()
                point = id_to_corners[marker_id][corner_index]
            ordered_points.append(point)

        ordered = np.array(ordered_points, dtype=np.float32)

        if not self._is_valid_quad(ordered):
            return None

        for idx, p in enumerate(ordered_points):
            px, py = int(p[0]), int(p[1])
            cv2.circle(frame, (px, py), 6, (0, 200, 255), -1)
            cv2.putText(
                frame,
                str(idx),
                (px + 6, py + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 255),
                1,
                cv2.LINE_AA,
            )

        return ordered

    def _world_quad(self):
        return np.array(
            [
                [self.world_bl_x.value(), self.world_bl_y.value()],
                [self.world_br_x.value(), self.world_br_y.value()],
                [self.world_tr_x.value(), self.world_tr_y.value()],
                [self.world_tl_x.value(), self.world_tl_y.value()],
            ],
            dtype=np.float32,
        )

    def _roi_destination_points(self, width: int, height: int):
        # Pixel coordinate system has origin at top-left.
        # We map world BL,BR,TR,TL to image BL,BR,TR,TL points.
        return np.array(
            [
                [0, height - 1],
                [width - 1, height - 1],
                [width - 1, 0],
                [0, 0],
            ],
            dtype=np.float32,
        )

    def _set_image(self, label: QLabel, bgr_frame) -> None:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(image)
        scaled = pix.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        label.setPixmap(scaled)

    def _on_roi_mouse_click(self, x: int, y: int) -> None:
        if self._latest_roi_size is None or self._latest_roi_to_world is None or not CV_AVAILABLE:
            return

        mapped = self._map_widget_to_image(self.roi_frame_label, QPoint(x, y), *self._latest_roi_size)
        if mapped is None:
            return

        rx, ry = mapped
        self.cursor_px_label.setText(f"{rx:.1f}, {ry:.1f}")

        point = np.array([[[rx, ry]]], dtype=np.float32)
        world = cv2.perspectiveTransform(point, self._latest_roi_to_world)[0][0]
        wx, wy = float(world[0]), float(world[1])
        self._cursor_world = (wx, wy)
        self.cursor_world_label.setText(f"X: {wx:.2f}  Y: {wy:.2f}")
        self.status_label.setText(f"Cursor captured at world X={wx:.2f}, Y={wy:.2f}")

    def _emit_move_request(self) -> None:
        if self._cursor_world is None:
            self.status_label.setText("Move target unavailable. Click on ROI first.")
            return

        wx, wy = self._cursor_world
        self.move_request_world_xy.emit(wx, wy, self.cursor_z_spin.value(), self.cursor_feed_spin.value())

    def _map_widget_to_image(
        self,
        label: QLabel,
        point: QPoint,
        img_width: int,
        img_height: int,
    ) -> Optional[tuple[float, float]]:
        w = label.width()
        h = label.height()
        if w <= 1 or h <= 1:
            return None

        sx = w / img_width
        sy = h / img_height
        scale = min(sx, sy)

        draw_w = img_width * scale
        draw_h = img_height * scale
        off_x = (w - draw_w) / 2.0
        off_y = (h - draw_h) / 2.0

        px = point.x()
        py = point.y()
        if px < off_x or py < off_y or px > off_x + draw_w or py > off_y + draw_h:
            return None

        ix = (px - off_x) / scale
        iy = (py - off_y) / scale
        if ix < 0 or iy < 0 or ix > img_width - 1 or iy > img_height - 1:
            return None

        return float(ix), float(iy)
