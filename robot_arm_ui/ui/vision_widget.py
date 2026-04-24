from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from robot_arm_ui.core.candy_detector import CandyDetector, Detection

from PyQt6.QtCore import QPoint, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QFont, QImage, QMouseEvent, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QColorDialog,
    QDialog,
)

try:
    import cv2  # type: ignore
    import numpy as np

    CV_AVAILABLE = True
except Exception:
    cv2 = None
    np = None
    CV_AVAILABLE = False


class ImageLabel(QLabel):
    mouse_clicked = pyqtSignal(int, int)

    def __init__(self, title: str) -> None:
        super().__init__(title)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(620, 420)
        self.setObjectName("VisionFrame")

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self.mouse_clicked.emit(event.position().toPoint().x(), event.position().toPoint().y())
        super().mousePressEvent(event)




class VisionWidget(QWidget):
    move_request_world_xy = pyqtSignal(float, float, float, float)
    gripper_request = pyqtSignal(str, float, float)
    # Kept for compatibility with existing main-window wiring.
    place_request_world = pyqtSignal(float, float, float, float, str, float, float)
    # Emits a list of dicts describing each step for the main window to convert & enqueue.
    pick_and_place_request = pyqtSignal(list)  # emits list[str] of G-code lines
    auto_sort_toggled = pyqtSignal(bool)       # emits True when sorting starts, False when stops
    save_settings_requested = pyqtSignal()
    load_settings_requested = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()

        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._update_frame)

        self._cap = None
        self._latest_roi_size: tuple[int, int] | None = None
        self._latest_roi_to_world = None
        self._cursor_world: tuple[float, float] | None = None
        self._last_marker_centers: dict[int, Any] = {}
        self._last_marker_corners: dict[int, Any] = {}

        # YOLO detection state
        self._detector = CandyDetector()
        self._detections: list[tuple[str, float, float, float]] = []  # (class, wx, wy, conf)

        # Default detection colors (BGR)
        self.det_colors: dict[int, tuple[int, int, int]] = {
            0: (168, 118, 49),
            1: (88, 190, 53),
            2: (47, 207, 245),
            3: (63, 44, 193),
            4: (164, 166, 13),
        }

        self._build_ui()
        self._connect_signals()
        self._create_default_files()

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
        self._cursor_world = None
        self._last_marker_centers.clear()
        self._last_marker_corners.clear()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(4, 4, 4, 4)
        content_layout.setSpacing(6)

        # Camera settings
        content_layout.addWidget(self._build_camera_group())

        # Frame view toggle
        frame_toggle_row = QHBoxLayout()
        frame_toggle_row.addWidget(QLabel("View:"))
        self.frame_view_combo = QComboBox()
        self.frame_view_combo.addItems(["ROI Only", "Original Only", "Both"])
        self.frame_view_combo.setCurrentIndex(0)  # ROI by default
        self.frame_view_combo.currentIndexChanged.connect(self._on_frame_view_changed)
        frame_toggle_row.addWidget(self.frame_view_combo)
        frame_toggle_row.addStretch(1)
        content_layout.addLayout(frame_toggle_row)

        # Camera frames
        self._frame_container = QWidget()
        self._frame_layout = QHBoxLayout(self._frame_container)
        self._frame_layout.setContentsMargins(0, 0, 0, 0)
        self.original_frame_label = ImageLabel("Original frame")
        self.roi_frame_label = ImageLabel("ROI frame (click to pick XY)")
        self._frame_layout.addWidget(self.original_frame_label, stretch=1)
        self._frame_layout.addWidget(self.roi_frame_label, stretch=1)
        content_layout.addWidget(self._frame_container)
        # Apply default view
        self._on_frame_view_changed(0)

        # --- Control panels (same order as numbering) ---
        content_layout.addWidget(self._collapsible(self._build_pick_group(), expanded=True))
        content_layout.addWidget(self._collapsible(self._build_move_group(), expanded=False))
        content_layout.addWidget(self._collapsible(self._build_gripper_group(), expanded=True))
        content_layout.addWidget(self._collapsible(self._build_places_group(), expanded=True))
        content_layout.addWidget(self._collapsible(self._build_pick_and_place_group(), expanded=True))
        content_layout.addWidget(self._collapsible(self._build_detection_group(), expanded=True))
        content_layout.addStretch(1)

        root.addWidget(self._wrap_scroll(content))

    def _on_frame_view_changed(self, index: int) -> None:
        """Toggle which camera frames are visible."""
        # 0=ROI Only, 1=Original Only, 2=Both
        self.original_frame_label.setVisible(index in (1, 2))
        self.roi_frame_label.setVisible(index in (0, 2))

    @staticmethod
    def _collapsible(group: QGroupBox, expanded: bool = True) -> QGroupBox:
        """Make a QGroupBox collapsible via its checkable toggle."""
        group.setCheckable(True)
        group.setChecked(expanded)
        # Hide/show children when toggled
        def _toggle(checked: bool) -> None:
            for child in group.findChildren(QWidget):
                if child.parent() == group:
                    child.setVisible(checked)
        group.toggled.connect(_toggle)
        if not expanded:
            _toggle(False)
        return group

    def _wrap_scroll(self, content: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QScrollArea.Shape.NoFrame)
        area.setWidget(content)
        return area

    def _build_camera_group(self) -> QGroupBox:
        group = QGroupBox("Vision")
        layout = QGridLayout(group)

        self.camera_index_spin = QSpinBox()
        self.camera_index_spin.setRange(0, 10)
        self.camera_index_spin.setValue(0)

        self.start_button = QPushButton("Start Camera")
        self.stop_button = QPushButton("Stop Camera")
        self.show_ids_check = QCheckBox("Show marker IDs")
        self.show_ids_check.setChecked(True)

        self.settings_toggle = QCheckBox("Show advanced settings")
        self.settings_toggle.setChecked(False)
        self.save_settings_button = QPushButton("Save Settings")
        self.load_settings_button = QPushButton("Load Settings")

        self.roi_width_spin = QSpinBox()
        self.roi_width_spin.setRange(200, 1600)
        self.roi_width_spin.setValue(700)

        self.roi_height_spin = QSpinBox()
        self.roi_height_spin.setRange(200, 1200)
        self.roi_height_spin.setValue(500)

        self.id_bl_spin = QSpinBox(); self.id_bl_spin.setRange(0, 49); self.id_bl_spin.setValue(0)
        self.id_br_spin = QSpinBox(); self.id_br_spin.setRange(0, 49); self.id_br_spin.setValue(1)
        self.id_tr_spin = QSpinBox(); self.id_tr_spin.setRange(0, 49); self.id_tr_spin.setValue(2)
        self.id_tl_spin = QSpinBox(); self.id_tl_spin.setRange(0, 49); self.id_tl_spin.setValue(3)

        self.anchor_mode_combo = QComboBox()
        self.anchor_mode_combo.addItems(["Center", "Corner"])

        self.corner_bl_combo = QComboBox()
        self.corner_br_combo = QComboBox()
        self.corner_tr_combo = QComboBox()
        self.corner_tl_combo = QComboBox()
        for combo in (self.corner_bl_combo, self.corner_br_combo, self.corner_tr_combo, self.corner_tl_combo):
            combo.addItems(["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"])

        self.corner_bl_combo.setCurrentIndex(1)
        self.corner_br_combo.setCurrentIndex(0)
        self.corner_tr_combo.setCurrentIndex(3)
        self.corner_tl_combo.setCurrentIndex(2)

        self.world_bl_x = self._float_spin(0.0)
        self.world_bl_y = self._float_spin(0.0)
        self.world_br_x = self._float_spin(300.0)
        self.world_br_y = self._float_spin(0.0)
        self.world_tr_x = self._float_spin(300.0)
        self.world_tr_y = self._float_spin(300.0)
        self.world_tl_x = self._float_spin(0.0)
        self.world_tl_y = self._float_spin(300.0)

        layout.addWidget(QLabel("Camera index"), 0, 0)
        layout.addWidget(self.camera_index_spin, 0, 1)
        layout.addWidget(self.start_button, 0, 2)
        layout.addWidget(self.stop_button, 0, 3)
        layout.addWidget(self.show_ids_check, 0, 4)

        layout.addWidget(self.settings_toggle, 1, 0)
        layout.addWidget(self.save_settings_button, 1, 3)
        layout.addWidget(self.load_settings_button, 1, 4)

        self._advanced_widgets: list[QWidget] = []

        self.lbl_roi_w = QLabel("ROI width")
        self.lbl_roi_h = QLabel("ROI height")
        self.lbl_id_bl = QLabel("ID BL")
        self.lbl_id_br = QLabel("ID BR")
        self.lbl_id_tr = QLabel("ID TR")
        self.lbl_id_tl = QLabel("ID TL")
        self.lbl_anchor = QLabel("Anchor mode")
        self.lbl_corner_bl = QLabel("BL corner")
        self.lbl_corner_br = QLabel("BR corner")
        self.lbl_corner_tr = QLabel("TR corner")
        self.lbl_corner_tl = QLabel("TL corner")
        self.lbl_w_bl_x = QLabel("World BL X")
        self.lbl_w_bl_y = QLabel("World BL Y")
        self.lbl_w_br_x = QLabel("World BR X")
        self.lbl_w_br_y = QLabel("World BR Y")
        self.lbl_w_tr_x = QLabel("World TR X")
        self.lbl_w_tr_y = QLabel("World TR Y")
        self.lbl_w_tl_x = QLabel("World TL X")
        self.lbl_w_tl_y = QLabel("World TL Y")

        layout.addWidget(self.lbl_roi_w, 2, 0)
        layout.addWidget(self.roi_width_spin, 2, 1)
        layout.addWidget(self.lbl_roi_h, 2, 2)
        layout.addWidget(self.roi_height_spin, 2, 3)

        layout.addWidget(self.lbl_id_bl, 3, 0)
        layout.addWidget(self.id_bl_spin, 3, 1)
        layout.addWidget(self.lbl_id_br, 3, 2)
        layout.addWidget(self.id_br_spin, 3, 3)
        layout.addWidget(self.lbl_id_tr, 4, 0)
        layout.addWidget(self.id_tr_spin, 4, 1)
        layout.addWidget(self.lbl_id_tl, 4, 2)
        layout.addWidget(self.id_tl_spin, 4, 3)

        layout.addWidget(self.lbl_anchor, 5, 0)
        layout.addWidget(self.anchor_mode_combo, 5, 1)
        layout.addWidget(self.lbl_corner_bl, 5, 2)
        layout.addWidget(self.corner_bl_combo, 5, 3)
        layout.addWidget(self.lbl_corner_br, 6, 0)
        layout.addWidget(self.corner_br_combo, 6, 1)
        layout.addWidget(self.lbl_corner_tr, 6, 2)
        layout.addWidget(self.corner_tr_combo, 6, 3)
        layout.addWidget(self.lbl_corner_tl, 7, 0)
        layout.addWidget(self.corner_tl_combo, 7, 1)

        layout.addWidget(self.lbl_w_bl_x, 8, 0)
        layout.addWidget(self.world_bl_x, 8, 1)
        layout.addWidget(self.lbl_w_bl_y, 8, 2)
        layout.addWidget(self.world_bl_y, 8, 3)

        layout.addWidget(self.lbl_w_br_x, 9, 0)
        layout.addWidget(self.world_br_x, 9, 1)
        layout.addWidget(self.lbl_w_br_y, 9, 2)
        layout.addWidget(self.world_br_y, 9, 3)

        layout.addWidget(self.lbl_w_tr_x, 10, 0)
        layout.addWidget(self.world_tr_x, 10, 1)
        layout.addWidget(self.lbl_w_tr_y, 10, 2)
        layout.addWidget(self.world_tr_y, 10, 3)

        layout.addWidget(self.lbl_w_tl_x, 11, 0)
        layout.addWidget(self.world_tl_x, 11, 1)
        layout.addWidget(self.lbl_w_tl_y, 11, 2)
        layout.addWidget(self.world_tl_y, 11, 3)

        self._advanced_widgets.extend(
            [
                self.lbl_roi_w, self.roi_width_spin, self.lbl_roi_h, self.roi_height_spin,
                self.lbl_id_bl, self.id_bl_spin, self.lbl_id_br, self.id_br_spin,
                self.lbl_id_tr, self.id_tr_spin, self.lbl_id_tl, self.id_tl_spin,
                self.lbl_anchor, self.anchor_mode_combo,
                self.lbl_corner_bl, self.corner_bl_combo,
                self.lbl_corner_br, self.corner_br_combo,
                self.lbl_corner_tr, self.corner_tr_combo,
                self.lbl_corner_tl, self.corner_tl_combo,
                self.lbl_w_bl_x, self.world_bl_x, self.lbl_w_bl_y, self.world_bl_y,
                self.lbl_w_br_x, self.world_br_x, self.lbl_w_br_y, self.world_br_y,
                self.lbl_w_tr_x, self.world_tr_x, self.lbl_w_tr_y, self.world_tr_y,
                self.lbl_w_tl_x, self.world_tl_x, self.lbl_w_tl_y, self.world_tl_y,
            ]
        )

        self.status_label = QLabel("Waiting for camera")
        layout.addWidget(self.status_label, 12, 0, 1, 6)

        self._set_advanced_settings_visible(self.settings_toggle.isChecked())
        return group

    def _build_pick_group(self) -> QGroupBox:
        group = QGroupBox("1) Pick XY From ROI")
        layout = QFormLayout(group)

        self.cursor_world_label = QLabel("X: --  Y: --")
        self.use_cursor_button = QPushButton("Use Cursor XY For Target")
        self.use_cursor_button.setStyleSheet("background: #2d8f5a; color: #ffffff;")

        layout.addRow("Cursor World", self.cursor_world_label)
        layout.addRow(self.use_cursor_button)
        return group

    def _build_move_group(self) -> QGroupBox:
        group = QGroupBox("2) Move / Jog")
        layout = QGridLayout(group)

        self.target_x_spin = self._float_spin(0.0)
        self.target_y_spin = self._float_spin(0.0)
        self.target_z_spin = self._float_spin(0.0)
        self.feed_spin = self._float_spin(30.0)

        self.move_target_button = QPushButton("Move To XYZ")
        self.move_target_button.setStyleSheet("background: #2b72d6; color: #ffffff;")

        self.jog_x_step_spin = self._float_spin(5.0)
        self.jog_y_step_spin = self._float_spin(5.0)
        self.jog_z_step_spin = self._float_spin(5.0)

        self.jog_x_minus = QPushButton("X-")
        self.jog_x_plus = QPushButton("X+")
        self.jog_y_minus = QPushButton("Y-")
        self.jog_y_plus = QPushButton("Y+")
        self.jog_z_minus = QPushButton("Z-")
        self.jog_z_plus = QPushButton("Z+")

        layout.addWidget(QLabel("Target X"), 0, 0)
        layout.addWidget(self.target_x_spin, 0, 1)
        layout.addWidget(QLabel("Target Y"), 0, 2)
        layout.addWidget(self.target_y_spin, 0, 3)
        layout.addWidget(QLabel("Target Z"), 0, 4)
        layout.addWidget(self.target_z_spin, 0, 5)

        layout.addWidget(QLabel("Feed"), 1, 0)
        layout.addWidget(self.feed_spin, 1, 1)
        layout.addWidget(self.move_target_button, 1, 2)

        layout.addWidget(QLabel("Jog X step"), 2, 0)
        layout.addWidget(self.jog_x_step_spin, 2, 1)
        layout.addWidget(self.jog_x_minus, 2, 2)
        layout.addWidget(self.jog_x_plus, 2, 3)

        layout.addWidget(QLabel("Jog Y step"), 3, 0)
        layout.addWidget(self.jog_y_step_spin, 3, 1)
        layout.addWidget(self.jog_y_minus, 3, 2)
        layout.addWidget(self.jog_y_plus, 3, 3)

        layout.addWidget(QLabel("Jog Z step"), 4, 0)
        layout.addWidget(self.jog_z_step_spin, 4, 1)
        layout.addWidget(self.jog_z_minus, 4, 2)
        layout.addWidget(self.jog_z_plus, 4, 3)

        return group

    def _build_gripper_group(self) -> QGroupBox:
        group = QGroupBox("3) Gripper")
        layout = QGridLayout(group)

        # Gripper distance controls
        self.gripper_close_dist_spin = self._float_spin(5.0)
        self.gripper_open_dist_spin = self._float_spin(20.0)
        self.gripper_feed_spin = self._float_spin(4.0)

        self.gripper_open_button = QPushButton("Open")
        self.gripper_close_button = QPushButton("Close")
        self.gripper_home_button = QPushButton("Home (M6)")
        self.gripper_open_button.setStyleSheet("background: #2d8f5a; color: #ffffff;")
        self.gripper_close_button.setStyleSheet("background: #b34a4a; color: #ffffff;")
        self.gripper_home_button.setStyleSheet("background: #4263eb; color: #ffffff;")

        layout.addWidget(QLabel("Close (mm)"), 0, 0)
        layout.addWidget(self.gripper_close_dist_spin, 0, 1)
        layout.addWidget(QLabel("Open (mm)"), 0, 2)
        layout.addWidget(self.gripper_open_dist_spin, 0, 3)
        layout.addWidget(QLabel("Speed"), 0, 4)
        layout.addWidget(self.gripper_feed_spin, 0, 5)
        layout.addWidget(self.gripper_open_button, 1, 0, 1, 2)
        layout.addWidget(self.gripper_close_button, 1, 2, 1, 2)
        layout.addWidget(self.gripper_home_button, 1, 4, 1, 2)
        return group

    def _build_places_group(self) -> QGroupBox:
        group = QGroupBox("4) Place Positions")
        layout = QVBoxLayout(group)

        top = QGridLayout()
        self.place_name_input = QLineEdit("P1")
        self.place_x_spin = self._float_spin(350.0)
        self.place_y_spin = self._float_spin(0.0)
        self.place_z_spin = self._float_spin(20.0)

        top.addWidget(QLabel("Name"), 0, 0)
        top.addWidget(self.place_name_input, 0, 1)
        top.addWidget(QLabel("X"), 0, 2)
        top.addWidget(self.place_x_spin, 0, 3)
        top.addWidget(QLabel("Y"), 0, 4)
        top.addWidget(self.place_y_spin, 0, 5)
        top.addWidget(QLabel("Z"), 0, 6)
        top.addWidget(self.place_z_spin, 0, 7)

        btn_row = QHBoxLayout()
        self.place_add_button = QPushButton("Add")
        self.place_update_button = QPushButton("Update")
        self.place_remove_button = QPushButton("Remove")
        self.place_load_button = QPushButton("Load Selected")
        self.place_move_button = QPushButton("Move Selected")
        self.place_place_button = QPushButton("Place Selected")
        self.place_place_button.setStyleSheet("background: #9a7a2f; color: #ffffff;")
        self.place_save_button = QPushButton("Save Places")
        self.place_load_file_button = QPushButton("Load Places")

        for b in (
            self.place_add_button,
            self.place_update_button,
            self.place_remove_button,
            self.place_load_button,
            self.place_move_button,
            self.place_place_button,
            self.place_save_button,
            self.place_load_file_button,
        ):
            btn_row.addWidget(b)

        self.place_table = QTableWidget(0, 4)
        self.place_table.setHorizontalHeaderLabels(["Name", "X", "Y", "Z"])
        self.place_table.verticalHeader().setVisible(False)
        self.place_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.place_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.place_table.setMinimumHeight(200)

        layout.addLayout(top)
        layout.addLayout(btn_row)
        layout.addWidget(self.place_table)
        return group

    def _connect_signals(self) -> None:
        self.start_button.clicked.connect(self._start_camera)
        self.stop_button.clicked.connect(self.stop)
        self.roi_frame_label.mouse_clicked.connect(self._on_roi_mouse_click)

        self.settings_toggle.toggled.connect(self._set_advanced_settings_visible)
        self.roi_width_spin.valueChanged.connect(self._sync_roi_display_width)
        self.save_settings_button.clicked.connect(self.save_settings_requested.emit)
        self.load_settings_button.clicked.connect(self.load_settings_requested.emit)

        self.use_cursor_button.clicked.connect(self._use_cursor_xy)
        self.move_target_button.clicked.connect(self._move_target)

        self.jog_x_minus.clicked.connect(lambda: self._jog_target(-self.jog_x_step_spin.value(), 0.0, 0.0))
        self.jog_x_plus.clicked.connect(lambda: self._jog_target(self.jog_x_step_spin.value(), 0.0, 0.0))
        self.jog_y_minus.clicked.connect(lambda: self._jog_target(0.0, -self.jog_y_step_spin.value(), 0.0))
        self.jog_y_plus.clicked.connect(lambda: self._jog_target(0.0, self.jog_y_step_spin.value(), 0.0))
        self.jog_z_minus.clicked.connect(lambda: self._jog_target(0.0, 0.0, -self.jog_z_step_spin.value()))
        self.jog_z_plus.clicked.connect(lambda: self._jog_target(0.0, 0.0, self.jog_z_step_spin.value()))

        self.gripper_open_button.clicked.connect(
            lambda: self.gripper_request.emit(
                "open",
                float(self.gripper_open_dist_spin.value()),
                float(self.gripper_feed_spin.value()),
            )
        )
        self.gripper_close_button.clicked.connect(
            lambda: self.gripper_request.emit(
                "close",
                float(self.gripper_close_dist_spin.value()),
                float(self.gripper_feed_spin.value()),
            )
        )
        self.gripper_home_button.clicked.connect(
            lambda: self.gripper_request.emit(
                "home",
                0.0,
                float(self.gripper_feed_spin.value()),
            )
        )

        self.place_add_button.clicked.connect(self._add_place_row)
        self.place_update_button.clicked.connect(self._update_place_row)
        self.place_remove_button.clicked.connect(self._remove_place_row)
        self.place_load_button.clicked.connect(self._load_selected_place)
        self.place_move_button.clicked.connect(self._move_selected_place)
        self.place_place_button.clicked.connect(self._place_selected_sequence)
        self.place_save_button.clicked.connect(self._save_places_file)
        self.place_load_file_button.clicked.connect(self._load_places_file)

        # Pick & Place sequence
        self.pnp_use_cursor_button.clicked.connect(self._pnp_use_cursor_xy)
        self.pnp_generate_button.clicked.connect(self._pnp_generate_default)
        self.pnp_save_template_button.clicked.connect(self._pnp_save_template)
        self.pnp_load_template_button.clicked.connect(self._pnp_load_template)
        self.pnp_execute_button.clicked.connect(self._pnp_execute)
        self.place_table.model().rowsInserted.connect(self._pnp_sync_place_combo)
        self.place_table.model().rowsRemoved.connect(self._pnp_sync_place_combo)
        self.place_table.model().dataChanged.connect(self._pnp_sync_place_combo)

        # Detection
        self.detect_load_button.clicked.connect(self._detect_load_model)
        self.detect_auto_pick_button.clicked.connect(self._on_auto_sort_clicked)

    def _set_advanced_settings_visible(self, visible: bool) -> None:
        for widget in self._advanced_widgets:
            widget.setVisible(visible)

    def _sync_roi_display_width(self) -> None:
        self.roi_frame_label.setMinimumWidth(max(300, self.roi_width_spin.value()))

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
        cap = cv2.VideoCapture(self.camera_index_spin.value())
        if not cap.isOpened():
            self.status_label.setText("Cannot open camera")
            return
        self._cap = cap
        self.status_label.setText("Camera running")
        self._timer.start()

    def _get_app_data_dir(self, subfolder: str = "") -> Path:
        docs = Path.home() / "Documents"
        base = docs if docs.exists() else Path.home()
        path = base / "Community-Robot-Arm-UI"
        if subfolder:
            path = path / subfolder
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _create_default_files(self) -> None:
        """Ensure default templates exist in the user's Documents folder on first run."""
        # 1. Default Colors
        colors_dir = self._get_app_data_dir("colors")
        default_colors_file = colors_dir / "default_candy_colors.json"
        if not default_colors_file.exists():
            default_colors = {
                "0": [168, 118, 49],
                "1": [88, 190, 53],
                "2": [47, 207, 245],
                "3": [63, 44, 193],
                "4": [164, 166, 13]
            }
            default_colors_file.write_text(json.dumps(default_colors, indent=4), encoding="utf-8")

        # 2. Default PnP Template
        templates_dir = self._get_app_data_dir("templates")
        default_pnp_file = templates_dir / "default_pnp.gcode"
        if not default_pnp_file.exists():
            lines = [
                "G90                                                     ; absolute mode",
                "G1 X{PICK_X} Y{PICK_Y} Z{APPROACH_Z} F{MOVE_FEED}      ; approach above pick",
                "M6                                                      ; gripper home",
                "G1 X{PICK_X} Y{PICK_Y} Z{PICK_Z} F{MOVE_FEED}          ; descend to pick",
                "M3 S{GRIP_CLOSE} F{GRIP_FEED}                           ; gripper close",
                "G1 X{PICK_X} Y{PICK_Y} Z{APPROACH_Z} F{PICK_FEED}      ; retract up from pick",
                "G1 X{PLACE_X} Y{PLACE_Y} Z{APPROACH_Z} F{PICK_FEED}    ; approach above place",
                "G1 X{PLACE_X} Y{PLACE_Y} Z{PLACE_Z} F{PICK_FEED}       ; descend to place",
                "M5 S{GRIP_OPEN} F{GRIP_FEED}                            ; gripper open",
                "G1 X{PLACE_X} Y{PLACE_Y} Z{APPROACH_Z} F{MOVE_FEED}    ; retract up from place",
            ]
            default_pnp_file.write_text("\n".join(lines), encoding="utf-8")

    def _update_frame(self) -> None:
        if self._cap is None:
            return

        ok, frame = self._cap.read()
        if not ok:
            self.status_label.setText("Camera read failed")
            return

        overlay = frame.copy()
        ordered = self._detect_ordered_corners(overlay)
        
        roi = None
        world_quad = None
        h_src_to_world = None
        h_src_to_roi = None
        polygon = None

        if ordered is None:
            self._latest_roi_size = None
            self._latest_roi_to_world = None
            self.roi_frame_label.setText("ROI frame\n(need 4 anchors)")
            self.status_label.setText("ROI lost")
        else:
            polygon = ordered.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(overlay, [polygon], isClosed=True, color=(0, 255, 255), thickness=2)

            roi_w = self.roi_width_spin.value()
            roi_h = self.roi_height_spin.value()
            dst = self._roi_destination_points(roi_w, roi_h)

            h_src_to_roi = cv2.getPerspectiveTransform(ordered.astype(np.float32), dst.astype(np.float32))
            roi = cv2.warpPerspective(frame, h_src_to_roi, (roi_w, roi_h))

            self._latest_roi_size = (roi_w, roi_h)
            world_quad = self._world_quad()
            self._latest_roi_to_world = cv2.getPerspectiveTransform(dst.astype(np.float32), world_quad.astype(np.float32))
            h_src_to_world = cv2.getPerspectiveTransform(ordered.astype(np.float32), world_quad.astype(np.float32))
            
            self.status_label.setText("ROI detected")

        # --- YOLO detection overlay ---
        self._detections = []
        if self._detector.is_loaded and self.detect_enable_check.isChecked():
            require_roi = self.detect_require_roi_check.isChecked()
            
            if (ordered is not None) or (not require_roi):
                cls_filter = None
                filter_text = self.detect_class_combo.currentText()
                if filter_text != "All Classes":
                    for i, name in enumerate(self._detector.class_names):
                        if name == filter_text:
                            cls_filter = i
                            break

                conf = self.detect_conf_spin.value()
                dets = self._detector.detect(frame, conf=conf, class_filter=cls_filter)

                for d in dets:
                    if polygon is not None:
                        if cv2.pointPolygonTest(polygon, (d.cx, d.cy), False) < 0:
                            continue

                    color = self.det_colors.get(d.class_id, (0, 255, 255))
                    cv2.rectangle(overlay, (d.x1, d.y1), (d.x2, d.y2), color, 2)
                    cv2.circle(overlay, (int(d.cx), int(d.cy)), 5, color, -1)
                    label = f"{d.class_name} {d.confidence:.2f}"
                    cv2.putText(overlay, label, (d.x1, d.y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

                    if h_src_to_world is not None and h_src_to_roi is not None and roi is not None:
                        pt_src = np.array([[[d.cx, d.cy]]], dtype=np.float32)
                        world_pt = cv2.perspectiveTransform(pt_src, h_src_to_world)[0][0]
                        wx, wy = float(world_pt[0]), float(world_pt[1])
                        self._detections.append((d.class_name, wx, wy, d.confidence))

                        roi_pt = cv2.perspectiveTransform(pt_src, h_src_to_roi)[0][0]
                        rx, ry = int(roi_pt[0]), int(roi_pt[1])
                        cv2.circle(roi, (rx, ry), 5, color, -1)
                        cv2.putText(roi, d.class_name, (rx + 8, ry + 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

                n = len(self._detections) if polygon is not None else len(dets)
                self.detect_status_label.setText(f"{n} detection(s)")

        if roi is not None:
            self._set_image(self.roi_frame_label, roi)
        self._set_image(self.original_frame_label, overlay)

    def _detect_ordered_corners(self, frame) -> Optional[Any]:
        if not CV_AVAILABLE:
            return None

        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(frame)

        id_to_center: dict[int, Any] = {}
        id_to_corners: dict[int, Any] = {}
        detected_ids: set[int] = set()

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for marker_corners, marker_id in zip(corners, ids.flatten().tolist()):
                marker_id = int(marker_id)
                pts = marker_corners.reshape(4, 2).astype(np.float32)
                center = pts.mean(axis=0).astype(np.float32)
                id_to_center[marker_id] = center
                id_to_corners[marker_id] = pts
                detected_ids.add(marker_id)
                self._last_marker_centers[marker_id] = center
                self._last_marker_corners[marker_id] = pts

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

        use_center = self.anchor_mode_combo.currentText().lower() == "center"
        ordered_points = []
        for marker_id, corner_combo in zip(ordered_ids, corner_combos):
            if marker_id in id_to_center:
                if use_center:
                    point = id_to_center[marker_id]
                else:
                    point = id_to_corners[marker_id][corner_combo.currentIndex()]
            elif marker_id in self._last_marker_centers:
                if use_center:
                    point = self._last_marker_centers[marker_id]
                elif marker_id in self._last_marker_corners:
                    point = self._last_marker_corners[marker_id][corner_combo.currentIndex()]
                else:
                    return None
            else:
                return None
            ordered_points.append(point)

        ordered = np.array(ordered_points, dtype=np.float32)
        if not self._is_valid_quad(ordered):
            return None

        for idx, (marker_id, p) in enumerate(zip(ordered_ids, ordered_points)):
            px, py = int(p[0]), int(p[1])
            color = (0, 200, 255) if marker_id in detected_ids else (0, 0, 255)
            label = str(idx) if marker_id in detected_ids else f"{idx}(S)"
            cv2.circle(frame, (px, py), 6, color, -1)
            cv2.putText(frame, label, (px + 6, py + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        return ordered

    def _is_valid_quad(self, ordered) -> bool:
        if ordered is None:
            return False
        quad = ordered.astype(np.float32)
        if np.any(~np.isfinite(quad)):
            return False
        area = abs(cv2.contourArea(quad.reshape((-1, 1, 2))))
        return area >= 100.0

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

        lw = max(1, label.width())
        lh = max(1, label.height())
        scaled = pix.scaled(lw, lh, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

        bg = QPixmap(lw, lh)
        bg.fill(Qt.GlobalColor.black)
        painter = QPainter(bg)
        painter.drawPixmap((lw - scaled.width()) // 2, (lh - scaled.height()) // 2, scaled)
        painter.end()
        label.setPixmap(bg)

    def _on_roi_mouse_click(self, x: int, y: int) -> None:
        if self._latest_roi_size is None or self._latest_roi_to_world is None or not CV_AVAILABLE:
            return
        mapped = self._map_widget_to_image(self.roi_frame_label, QPoint(x, y), *self._latest_roi_size)
        if mapped is None:
            return

        rx, ry = mapped

        # If detection is active, snap to nearest centroid
        if (
            self._detector.is_loaded
            and self.detect_enable_check.isChecked()
            and self._detections
        ):
            # Re-run centroid→pixel by inverse: we have world coords in _detections,
            # but we can simply compare with stored pixel centroid by re-transforming.
            # Simpler: just use world coords and pick the nearest detection.
            point = np.array([[[rx, ry]]], dtype=np.float32)
            world_click = cv2.perspectiveTransform(point, self._latest_roi_to_world)[0][0]
            click_wx, click_wy = float(world_click[0]), float(world_click[1])

            best_dist = float('inf')
            best_det = None
            for name, wx, wy, conf in self._detections:
                dist = (wx - click_wx) ** 2 + (wy - click_wy) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_det = (name, wx, wy, conf)

            # Snap threshold: 30mm in world space
            if best_det is not None and best_dist < 30.0 ** 2:
                name, wx, wy, conf = best_det
                self._cursor_world = (wx, wy)
                self.cursor_world_label.setText(f"X: {wx:.2f}  Y: {wy:.2f}")
                self.pnp_pick_x_spin.setValue(wx)
                self.pnp_pick_y_spin.setValue(wy)
                self.status_label.setText(
                    f"Picked detection: {name} ({conf:.0%}) → X={wx:.2f}, Y={wy:.2f}"
                )
                return

        # Default: free-click world coordinate
        point = np.array([[[rx, ry]]], dtype=np.float32)
        world = cv2.perspectiveTransform(point, self._latest_roi_to_world)[0][0]
        wx, wy = float(world[0]), float(world[1])
        self._cursor_world = (wx, wy)
        self.cursor_world_label.setText(f"X: {wx:.2f}  Y: {wy:.2f}")
        self.status_label.setText(f"Cursor captured: X={wx:.2f}, Y={wy:.2f}")

    def _use_cursor_xy(self) -> None:
        if self._cursor_world is None:
            self.status_label.setText("Click ROI to capture XY first")
            return
        self.target_x_spin.setValue(self._cursor_world[0])
        self.target_y_spin.setValue(self._cursor_world[1])

    def _move_target(self) -> None:
        self.move_request_world_xy.emit(
            float(self.target_x_spin.value()),
            float(self.target_y_spin.value()),
            float(self.target_z_spin.value()),
            float(self.feed_spin.value()),
        )

    def _jog_target(self, dx: float, dy: float, dz: float) -> None:
        self.target_x_spin.setValue(self.target_x_spin.value() + dx)
        self.target_y_spin.setValue(self.target_y_spin.value() + dy)
        self.target_z_spin.setValue(self.target_z_spin.value() + dz)
        self._move_target()

    def _map_widget_to_image(self, label: QLabel, point: QPoint, img_width: int, img_height: int) -> Optional[tuple[float, float]]:
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

    def _selected_place_row(self) -> int | None:
        selected = self.place_table.selectionModel().selectedRows()
        if not selected:
            return None
        return int(selected[0].row())

    def _set_place_row(self, row: int, name: str, x: float, y: float, z: float) -> None:
        self.place_table.setItem(row, 0, QTableWidgetItem(name))
        self.place_table.setItem(row, 1, QTableWidgetItem(f"{x:.3f}"))
        self.place_table.setItem(row, 2, QTableWidgetItem(f"{y:.3f}"))
        self.place_table.setItem(row, 3, QTableWidgetItem(f"{z:.3f}"))

    def _row_values(self, row: int) -> Optional[tuple[str, float, float, float]]:
        try:
            name = self.place_table.item(row, 0).text()
            x = float(self.place_table.item(row, 1).text())
            y = float(self.place_table.item(row, 2).text())
            z = float(self.place_table.item(row, 3).text())
            return name, x, y, z
        except Exception:
            return None

    def _add_place_row(self) -> None:
        row = self.place_table.rowCount()
        self.place_table.insertRow(row)
        self._set_place_row(
            row,
            self.place_name_input.text().strip() or f"P{row + 1}",
            self.place_x_spin.value(),
            self.place_y_spin.value(),
            self.place_z_spin.value(),
        )

    def _update_place_row(self) -> None:
        row = self._selected_place_row()
        if row is None:
            self.status_label.setText("Select place row to update")
            return
        self._set_place_row(
            row,
            self.place_name_input.text().strip() or f"P{row + 1}",
            self.place_x_spin.value(),
            self.place_y_spin.value(),
            self.place_z_spin.value(),
        )

    def _remove_place_row(self) -> None:
        row = self._selected_place_row()
        if row is None:
            self.status_label.setText("Select place row to remove")
            return
        self.place_table.removeRow(row)

    def _load_selected_place(self) -> None:
        row = self._selected_place_row()
        if row is None:
            self.status_label.setText("Select place row to load")
            return
        values = self._row_values(row)
        if values is None:
            return
        name, x, y, z = values
        self.place_name_input.setText(name)
        self.place_x_spin.setValue(x)
        self.place_y_spin.setValue(y)
        self.place_z_spin.setValue(z)
        self.target_x_spin.setValue(x)
        self.target_y_spin.setValue(y)
        self.target_z_spin.setValue(z)

    def _move_selected_place(self) -> None:
        row = self._selected_place_row()
        if row is None:
            self.status_label.setText("Select place row to move")
            return
        values = self._row_values(row)
        if values is None:
            return
        _, x, y, z = values
        self.move_request_world_xy.emit(x, y, z, float(self.feed_spin.value()))

    def _place_selected_sequence(self) -> None:
        row = self._selected_place_row()
        if row is None:
            self.status_label.setText("Select place row first")
            return
        values = self._row_values(row)
        if values is None:
            return

        _, x, y, z = values

        self.place_request_world.emit(
            float(x),
            float(y),
            float(z),
            float(self.feed_spin.value()),
            "none",
            0.0,
            float(self.gripper_feed_spin.value()),
        )

    def _save_places_file(self) -> None:
        places_dir = self._get_app_data_dir("places")
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Save Place Positions",
            str(places_dir / "places.json"),
            "JSON Files (*.json);;All Files (*.*)",
        )
        if not path_str:
            return

        places: list[dict[str, Any]] = []
        for row in range(self.place_table.rowCount()):
            values = self._row_values(row)
            if values is None:
                continue
            name, x, y, z = values
            places.append({"name": name, "x": x, "y": y, "z": z})

        try:
            Path(path_str).write_text(json.dumps({"places": places}, indent=2), encoding="utf-8")
            self.status_label.setText(f"Saved places: {path_str}")
        except Exception as exc:
            self.status_label.setText(f"Save places failed: {exc}")

    def _load_places_file(self) -> None:
        places_dir = self._get_app_data_dir("places")
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Load Place Positions",
            str(places_dir),
            "JSON Files (*.json);;All Files (*.*)",
        )
        if not path_str:
            return

        try:
            payload = json.loads(Path(path_str).read_text(encoding="utf-8"))
            self.place_table.setRowCount(0)
            for place in payload.get("places", []):
                row = self.place_table.rowCount()
                self.place_table.insertRow(row)
                self._set_place_row(
                    row,
                    str(place.get("name", f"P{row + 1}")),
                    float(place.get("x", 0.0)),
                    float(place.get("y", 0.0)),
                    float(place.get("z", 0.0)),
                )
            self.status_label.setText(f"Loaded places: {path_str}")
        except Exception as exc:
            self.status_label.setText(f"Load places failed: {exc}")
        self._pnp_sync_place_combo()

    # ------------------------------------------------------------------
    # Pick & Place – UI builder
    # ------------------------------------------------------------------

    def _build_pick_and_place_group(self) -> QGroupBox:
        group = QGroupBox("5) Pick & Place Sequence (G-code, world coords)")
        layout = QVBoxLayout(group)

        # --- Input parameters (used by "Generate Default") ---
        params = QGridLayout()

        self.pnp_pick_x_spin = self._float_spin(0.0)
        self.pnp_pick_y_spin = self._float_spin(0.0)
        self.pnp_pick_z_spin = self._float_spin(0.0)
        self.pnp_approach_z_spin = self._float_spin(50.0)
        self.pnp_move_feed_spin = self._float_spin(30.0)
        self.pnp_pick_feed_spin = self._float_spin(10.0)

        self.pnp_use_cursor_button = QPushButton("Use Cursor XY for Pick")
        self.pnp_use_cursor_button.setStyleSheet("background: #2d8f5a; color: #ffffff;")

        self.pnp_place_combo = QComboBox()
        self.pnp_place_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        params.addWidget(QLabel("Pick X"), 0, 0)
        params.addWidget(self.pnp_pick_x_spin, 0, 1)
        params.addWidget(QLabel("Pick Y"), 0, 2)
        params.addWidget(self.pnp_pick_y_spin, 0, 3)
        params.addWidget(QLabel("Pick Z"), 0, 4)
        params.addWidget(self.pnp_pick_z_spin, 0, 5)
        params.addWidget(self.pnp_use_cursor_button, 0, 6)

        params.addWidget(QLabel("Approach Z"), 1, 0)
        params.addWidget(self.pnp_approach_z_spin, 1, 1)
        params.addWidget(QLabel("Move Feed"), 1, 2)
        params.addWidget(self.pnp_move_feed_spin, 1, 3)
        params.addWidget(QLabel("Pick Feed"), 1, 4)
        params.addWidget(self.pnp_pick_feed_spin, 1, 5)

        params.addWidget(QLabel("Place"), 2, 0)
        params.addWidget(self.pnp_place_combo, 2, 1, 1, 6)

        layout.addLayout(params)

        # --- G-code sequence editor ---
        self.pnp_gcode_edit = QPlainTextEdit()
        self.pnp_gcode_edit.setPlaceholderText(
            "G-code template (world coordinates).\n"
            "Lines starting with ; are comments.\n\n"
            "Available variables (resolved at execution time):\n"
            "  {PICK_X} {PICK_Y} {PICK_Z}     — from Pick fields above\n"
            "  {PLACE_X} {PLACE_Y} {PLACE_Z}   — from selected Place\n"
            "  {APPROACH_Z}                     — from Approach Z field\n"
            "  {MOVE_FEED}                      — from Move Feed field (fast travel)\n"
            "  {PICK_FEED}                      — from Pick Feed field (slow pick/place)\n"
            "  {GRIP_CLOSE} {GRIP_OPEN}         — from Gripper dist fields\n"
            "  {GRIP_FEED}                      — from Gripper feed field\n\n"
            "Example:\n"
            "  G1 X{PICK_X} Y{PICK_Y} Z{APPROACH_Z} F{MOVE_FEED}\n"
            "  G1 X{PICK_X} Y{PICK_Y} Z{PICK_Z} F{PICK_FEED}\n"
            "  M3 S{GRIP_CLOSE} F{GRIP_FEED}"
        )
        self.pnp_gcode_edit.setMinimumHeight(300)
        self.pnp_gcode_edit.setFont(QFont("Consolas", 10))
        layout.addWidget(self.pnp_gcode_edit)

        # --- Buttons ---
        btn_row = QHBoxLayout()
        self.pnp_generate_button = QPushButton("Generate Default")
        self.pnp_save_template_button = QPushButton("Save Template")
        self.pnp_load_template_button = QPushButton("Load Template")
        btn_row.addWidget(self.pnp_generate_button)
        btn_row.addWidget(self.pnp_save_template_button)
        btn_row.addWidget(self.pnp_load_template_button)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        exec_row = QHBoxLayout()
        self.pnp_execute_button = QPushButton("▶  Execute Pick & Place")
        self.pnp_execute_button.setStyleSheet(
            "background: #b35c00; color: #ffffff; font-weight: bold; padding: 10px 18px;"
        )
        self.pnp_stop_button = QPushButton("⏹  Stop")
        self.pnp_stop_button.setStyleSheet(
            "background: #b34a4a; color: #ffffff; font-weight: bold; padding: 10px 18px;"
        )
        self.pnp_stop_button.setEnabled(False)
        exec_row.addWidget(self.pnp_execute_button)
        exec_row.addWidget(self.pnp_stop_button)
        layout.addLayout(exec_row)

        # Populate default sequence on first build
        self._pnp_generate_default()

        return group

    def _pnp_save_template(self) -> None:
        """Save the current PnP G-code template to a file."""
        templates_dir = self._get_app_data_dir("templates")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save PnP Template",
            str(templates_dir / "pnp_template.gcode"),
            "G-code Files (*.gcode *.txt);;All Files (*.*)",
        )
        if not path:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.pnp_gcode_edit.toPlainText(), encoding="utf-8")
        self.status_label.setText(f"Template saved: {path}")

    def _pnp_load_template(self) -> None:
        """Load a PnP G-code template from a file."""
        templates_dir = self._get_app_data_dir("templates")
        path, _ = QFileDialog.getOpenFileName(
            self, "Load PnP Template",
            str(templates_dir),
            "G-code Files (*.gcode *.txt);;All Files (*.*)",
        )
        if not path:
            return
        try:
            text = Path(path).read_text(encoding="utf-8")
            self.pnp_gcode_edit.setPlainText(text)
            self.status_label.setText(f"Template loaded: {path}")
        except Exception as exc:
            self.status_label.setText(f"Failed to load template: {exc}")

    # ------------------------------------------------------------------
    # Detection – UI builder
    # ------------------------------------------------------------------

    def _build_detection_group(self) -> QGroupBox:
        group = QGroupBox("6) YOLO Detection & Auto Sort")
        layout = QVBoxLayout(group)

        # --- Row 1: Model ---
        model_row = QHBoxLayout()
        self.detect_load_button = QPushButton("Load Model")
        self.detect_model_label = QLabel("No model loaded")
        model_row.addWidget(self.detect_load_button)
        model_row.addWidget(self.detect_model_label, stretch=1)
        layout.addLayout(model_row)

        # --- Row 2: Detection settings ---
        settings_row = QHBoxLayout()
        self.detect_enable_check = QCheckBox("Enable Detection")
        self.detect_enable_check.setChecked(False)
        self.detect_enable_check.setEnabled(False)  # until model loaded

        self.detect_require_roi_check = QCheckBox("Require ROI")
        self.detect_require_roi_check.setChecked(True)

        self.detect_conf_spin = QDoubleSpinBox()
        self.detect_conf_spin.setRange(0.05, 1.0)
        self.detect_conf_spin.setSingleStep(0.05)
        self.detect_conf_spin.setValue(0.5)
        self.detect_conf_spin.setDecimals(2)

        self.detect_class_combo = QComboBox()
        self.detect_class_combo.addItem("All Classes")

        self.detect_color_btn = QPushButton("Colors...")
        self.detect_color_btn.clicked.connect(self._edit_class_colors)

        settings_row.addWidget(self.detect_enable_check)
        settings_row.addWidget(self.detect_require_roi_check)
        settings_row.addWidget(QLabel("Confidence"))
        settings_row.addWidget(self.detect_conf_spin)
        settings_row.addWidget(QLabel("Class"))
        settings_row.addWidget(self.detect_class_combo)
        settings_row.addWidget(self.detect_color_btn)
        settings_row.addStretch(1)
        layout.addLayout(settings_row)

        # --- Row 3: Idle timeout + Wake delay ---
        timeout_row = QHBoxLayout()
        self.detect_idle_timeout_spin = QDoubleSpinBox()
        self.detect_idle_timeout_spin.setRange(0.5, 30.0)
        self.detect_idle_timeout_spin.setSingleStep(0.5)
        self.detect_idle_timeout_spin.setValue(1.0)
        self.detect_idle_timeout_spin.setDecimals(1)
        self.detect_idle_timeout_spin.setSuffix(" s")

        self.detect_wake_delay_spin = QDoubleSpinBox()
        self.detect_wake_delay_spin.setRange(0.0, 10.0)
        self.detect_wake_delay_spin.setSingleStep(0.1)
        self.detect_wake_delay_spin.setValue(0.5)
        self.detect_wake_delay_spin.setDecimals(1)
        self.detect_wake_delay_spin.setSuffix(" s")

        timeout_row.addWidget(QLabel("Idle Timeout"))
        timeout_row.addWidget(self.detect_idle_timeout_spin)
        timeout_row.addWidget(QLabel("Wake Delay"))
        timeout_row.addWidget(self.detect_wake_delay_spin)
        timeout_row.addStretch(1)
        layout.addLayout(timeout_row)

        # --- Park sequence (no detection → go to idle) ---
        layout.addWidget(QLabel("Park Sequence (sent when no detection after timeout):"))
        self.detect_park_edit = QPlainTextEdit()
        self.detect_park_edit.setPlainText(
            "G0 T10 T290 T30   ; park position\n"
            "M6                ; home gripper\n"
            "M18               ; disable motors"
        )
        self.detect_park_edit.setMaximumHeight(80)
        _mono = QFont("Consolas", 9)
        self.detect_park_edit.setFont(_mono)
        layout.addWidget(self.detect_park_edit)

        # --- Wake-up sequence (detection found after being parked) ---
        layout.addWidget(QLabel("Wake-up Sequence (sent when detection found after park):"))
        self.detect_wake_edit = QPlainTextEdit()
        self.detect_wake_edit.setPlainText(
            "M17               ; enable motors"
        )
        self.detect_wake_edit.setMaximumHeight(50)
        self.detect_wake_edit.setFont(_mono)
        layout.addWidget(self.detect_wake_edit)

        # --- Auto sort button + status ---
        sort_row = QHBoxLayout()
        self.detect_auto_pick_button = QPushButton("▶  Start Auto Sort")
        self.detect_auto_pick_button.setStyleSheet(
            "background: #6b2fa0; color: #ffffff; font-weight: bold; padding: 8px 14px;"
        )
        self.detect_auto_pick_button.setEnabled(False)
        self.detect_auto_pick_button.setCheckable(True)
        self.detect_auto_pick_button.setToolTip(
            "Start a continuous Look-Pick-Look cycle.\n"
            "Sorting trick: If you create a Place with the exact same name as a candy class "
            "(e.g., 'Lemon'), that candy will automatically be sorted to that place!"
        )

        self.detect_status_label = QLabel("—")
        sort_row.addWidget(self.detect_auto_pick_button)
        sort_row.addWidget(self.detect_status_label, stretch=1)
        layout.addLayout(sort_row)

        return group

    # ------------------------------------------------------------------
    # Detection – helpers
    # ------------------------------------------------------------------

    def _detect_load_model(self) -> None:
        """Open file picker and load an ONNX or YOLO .pt model."""
        from PyQt6.QtWidgets import QFileDialog, QMessageBox

        if not CandyDetector.is_available():
            QMessageBox.warning(
                self, "No Inference Backend",
                "No inference backend is available.\n\n"
                "Install one of:\n"
                "    pip install onnxruntime     (recommended, lightweight)\n"
                "    pip install ultralytics     (full YOLO framework)\n\n"
                "Then restart the application.",
            )
            return

        models_dir = self._get_app_data_dir("models")
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Detection Model",
            str(models_dir),
            CandyDetector.supported_formats(),
        )
        if not path:
            return

        try:
            self._detector.load(path)
        except Exception as exc:
            QMessageBox.critical(
                self, "Model Load Failed",
                f"Could not load the model:\n\n{exc}",
            )
            return

        name = Path(path).stem
        n_cls = len(self._detector.class_names)
        backend = "ONNX" if path.endswith(".onnx") else "YOLO"
        self.detect_model_label.setText(f"{name} · {n_cls} classes ({backend})")
        self.detect_enable_check.setEnabled(True)
        self.detect_enable_check.setChecked(True)
        self.detect_auto_pick_button.setEnabled(True)

        # Populate class filter combo
        self.detect_class_combo.clear()
        self.detect_class_combo.addItem("All Classes")
        for cls_name in self._detector.class_names:
            self.detect_class_combo.addItem(cls_name)

        self.status_label.setText(f"Model loaded ({backend}): {path}")

    def _edit_class_colors(self) -> None:
        """Open a dialog to edit class detection colors."""
        if not self._detector.class_names:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(self, "No Model", "Please load a model first to edit its class colors.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Class Colors")
        dialog.resize(320, 450)
        main_layout = QVBoxLayout(dialog)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content_widget = QWidget()
        layout = QFormLayout(content_widget)

        color_buttons = {}
        for i, name in enumerate(self._detector.class_names):
            b, g, r = self.det_colors.get(i, (0, 255, 255))
            btn = QPushButton()
            btn.setStyleSheet(f"background-color: rgb({r}, {g}, {b}); border: 1px solid #555;")
            btn.setFixedSize(60, 24)
            layout.addRow(QLabel(f"{i}: {name}"), btn)
            color_buttons[i] = btn

            def make_handler(cls_id=i, button=btn):
                def handler():
                    from PyQt6.QtGui import QColor
                    cb, cg, cr = self.det_colors.get(cls_id, (0, 255, 255))
                    init_color = QColor(cr, cg, cb)
                    c = QColorDialog.getColor(init_color, dialog, f"Select Color for {self._detector.class_names[cls_id]}")
                    if c.isValid():
                        self.det_colors[cls_id] = (c.blue(), c.green(), c.red())
                        button.setStyleSheet(f"background-color: rgb({c.red()}, {c.green()}, {c.blue()}); border: 1px solid #555;")
                return handler

            btn.clicked.connect(make_handler())

        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)

        # Buttons for Import / Export
        btn_layout = QHBoxLayout()
        export_btn = QPushButton("Export")
        import_btn = QPushButton("Import")
        close_btn = QPushButton("Done")
        
        btn_layout.addWidget(import_btn)
        btn_layout.addWidget(export_btn)
        btn_layout.addStretch(1)
        btn_layout.addWidget(close_btn)

        main_layout.addLayout(btn_layout)

        def do_export():
            colors_dir = self._get_app_data_dir("colors")
            path, _ = QFileDialog.getSaveFileName(dialog, "Export Colors", str(colors_dir / "theme.json"), "JSON Files (*.json)")
            if path:
                with open(path, "w") as f:
                    json.dump(self.det_colors, f, indent=4)

        def do_import():
            colors_dir = self._get_app_data_dir("colors")
            path, _ = QFileDialog.getOpenFileName(dialog, "Import Colors", str(colors_dir), "JSON Files (*.json)")
            if path:
                try:
                    with open(path, "r") as f:
                        data = json.load(f)
                    for k, v in data.items():
                        cls_id = int(k)
                        self.det_colors[cls_id] = (int(v[0]), int(v[1]), int(v[2]))
                        if cls_id in color_buttons:
                            b, g, r = self.det_colors[cls_id]
                            color_buttons[cls_id].setStyleSheet(f"background-color: rgb({r}, {g}, {b}); border: 1px solid #555;")
                except Exception as e:
                    from PyQt6.QtWidgets import QMessageBox
                    QMessageBox.warning(dialog, "Error", f"Failed to load colors: {e}")

        export_btn.clicked.connect(do_export)
        import_btn.clicked.connect(do_import)
        close_btn.clicked.connect(dialog.accept)

        dialog.exec()

    def _on_auto_sort_clicked(self, checked: bool) -> None:
        """Toggle the continuous auto-sort state."""
        if checked:
            self.detect_auto_pick_button.setText("⏹  Stop Auto Sort")
            self.detect_auto_pick_button.setStyleSheet("background: #b34a4a; color: #ffffff; font-weight: bold; padding: 8px 14px;")
            self.detect_status_label.setText("Auto Sort: Running")
        else:
            self.detect_auto_pick_button.setText("▶  Start Auto Sort")
            self.detect_auto_pick_button.setStyleSheet("background: #6b2fa0; color: #ffffff; font-weight: bold; padding: 8px 14px;")
            self.detect_status_label.setText("Auto Sort: Stopped")
        
        self.auto_sort_toggled.emit(checked)

    def get_best_detection(self) -> tuple[str, float, float] | None:
        """Return the (class_name, wx, wy) of the most confident detection in the current frame."""
        if not self._detections:
            return None
        
        # Sort by confidence descending
        sorted_dets = sorted(self._detections, key=lambda d: d[3], reverse=True)
        name, wx, wy, _conf = sorted_dets[0]
        return (name, wx, wy)

    # ------------------------------------------------------------------
    # Pick & Place – helpers
    # ------------------------------------------------------------------

    def _pnp_use_cursor_xy(self) -> None:
        if self._cursor_world is None:
            self.status_label.setText("Click ROI to capture cursor XY first")
            return
        self.pnp_pick_x_spin.setValue(self._cursor_world[0])
        self.pnp_pick_y_spin.setValue(self._cursor_world[1])
        self.status_label.setText(
            f"Pick XY set from cursor: {self._cursor_world[0]:.2f}, {self._cursor_world[1]:.2f}"
        )

    def _pnp_sync_place_combo(self) -> None:
        """Re-populate the place combo from the place table."""
        current = self.pnp_place_combo.currentText()
        self.pnp_place_combo.blockSignals(True)
        self.pnp_place_combo.clear()
        for row in range(self.place_table.rowCount()):
            vals = self._row_values(row)
            if vals:
                name, x, y, z = vals
                self.pnp_place_combo.addItem(
                    f"{name}  ({x:.1f}, {y:.1f}, {z:.1f})", row
                )
        if current:
            idx = self.pnp_place_combo.findText(current)
            if idx >= 0:
                self.pnp_place_combo.setCurrentIndex(idx)
        self.pnp_place_combo.blockSignals(False)

    def _pnp_selected_place(self) -> Optional[tuple[str, float, float, float]]:
        """Return (name, x, y, z) of the currently selected place, or None."""
        idx = self.pnp_place_combo.currentIndex()
        if idx < 0:
            return None
        row = self.pnp_place_combo.itemData(idx)
        if row is None or row >= self.place_table.rowCount():
            return None
        return self._row_values(row)

    def _pnp_build_variables(self) -> dict[str, str]:
        """Build the variable dictionary from current UI state.

        Returns a dict mapping variable names (without braces) to their
        string values.  Used both for template resolution and for
        documenting available variables.
        """
        place = self._pnp_selected_place()
        plx, ply, plz = (place[1], place[2], place[3]) if place else (0.0, 0.0, 0.0)

        def g(v: float) -> str:
            return f"{v:g}"

        return {
            "PICK_X": g(self.pnp_pick_x_spin.value()),
            "PICK_Y": g(self.pnp_pick_y_spin.value()),
            "PICK_Z": g(self.pnp_pick_z_spin.value()),
            "PLACE_X": g(plx),
            "PLACE_Y": g(ply),
            "PLACE_Z": g(plz),
            "APPROACH_Z": g(self.pnp_approach_z_spin.value()),
            "MOVE_FEED": g(self.pnp_move_feed_spin.value()),
            "PICK_FEED": g(self.pnp_pick_feed_spin.value()),
            "GRIP_CLOSE": g(self.gripper_close_dist_spin.value()),
            "GRIP_OPEN": g(self.gripper_open_dist_spin.value()),
            "GRIP_FEED": g(self.gripper_feed_spin.value()),
        }

    def _pnp_generate_default(self) -> None:
        """Generate the default pick-and-place G-code template with variables."""
        lines = [
            "G90                                                     ; absolute mode",
            "G1 X{PICK_X} Y{PICK_Y} Z{APPROACH_Z} F{MOVE_FEED}      ; approach above pick",
            "M6                                                      ; gripper home",
            "G1 X{PICK_X} Y{PICK_Y} Z{PICK_Z} F{MOVE_FEED}          ; descend to pick",
            "M3 S{GRIP_CLOSE} F{GRIP_FEED}                           ; gripper close",
            "G1 X{PICK_X} Y{PICK_Y} Z{APPROACH_Z} F{PICK_FEED}      ; retract up from pick",
            "G1 X{PLACE_X} Y{PLACE_Y} Z{APPROACH_Z} F{PICK_FEED}    ; approach above place",
            "G1 X{PLACE_X} Y{PLACE_Y} Z{PLACE_Z} F{PICK_FEED}       ; descend to place",
            "M5 S{GRIP_OPEN} F{GRIP_FEED}                            ; gripper open",
            "G1 X{PLACE_X} Y{PLACE_Y} Z{APPROACH_Z} F{MOVE_FEED}    ; retract up from place",
        ]

        self.pnp_gcode_edit.setPlainText("\n".join(lines))
        self.status_label.setText("Pick & Place: default template generated")

    def _pnp_execute(self) -> None:
        """Resolve variables, strip comments, and emit G-code for execution."""
        text = self.pnp_gcode_edit.toPlainText().strip()
        if not text:
            self.status_label.setText("Pick & Place: no G-code to execute")
            return

        # Build variable map from current UI state
        variables = self._pnp_build_variables()

        lines: list[str] = []
        for raw_line in text.splitlines():
            # Strip comments (everything after ;)
            cmd = raw_line.split(";")[0].strip()
            if not cmd:
                continue

            # Resolve {VAR} placeholders
            try:
                resolved = cmd.format_map(variables)
            except (KeyError, ValueError) as exc:
                self.status_label.setText(f"Pick & Place: variable error: {exc}")
                return

            lines.append(resolved)

        if not lines:
            self.status_label.setText("Pick & Place: no executable lines")
            return

        self.pick_and_place_request.emit(lines)
        self.status_label.setText(f"Pick & Place: {len(lines)} command(s) queued")

    # ------------------------------------------------------------------
    # Pick & Place – settings persistence
    # ------------------------------------------------------------------

    def _pnp_get_config(self) -> dict[str, Any]:
        return {
            "pick_x": self.pnp_pick_x_spin.value(),
            "pick_y": self.pnp_pick_y_spin.value(),
            "pick_z": self.pnp_pick_z_spin.value(),
            "approach_z": self.pnp_approach_z_spin.value(),
            "move_feed": self.pnp_move_feed_spin.value(),
            "pick_feed": self.pnp_pick_feed_spin.value(),
            "place_index": self.pnp_place_combo.currentIndex(),
            "gcode": self.pnp_gcode_edit.toPlainText(),
        }

    def _pnp_apply_config(self, cfg: dict[str, Any]) -> None:
        self.pnp_pick_x_spin.setValue(float(cfg.get("pick_x", 0.0)))
        self.pnp_pick_y_spin.setValue(float(cfg.get("pick_y", 0.0)))
        self.pnp_pick_z_spin.setValue(float(cfg.get("pick_z", 0.0)))
        self.pnp_approach_z_spin.setValue(float(cfg.get("approach_z", 50.0)))
        self.pnp_move_feed_spin.setValue(float(cfg.get("move_feed", 30.0)))
        self.pnp_pick_feed_spin.setValue(float(cfg.get("pick_feed", 10.0)))

        gcode = cfg.get("gcode", "")
        if gcode:
            self.pnp_gcode_edit.setPlainText(gcode)

        idx = int(cfg.get("place_index", 0))
        if 0 <= idx < self.pnp_place_combo.count():
            self.pnp_place_combo.setCurrentIndex(idx)

    def get_settings(self) -> dict[str, Any]:
        places: list[dict[str, Any]] = []
        for row in range(self.place_table.rowCount()):
            values = self._row_values(row)
            if values is None:
                continue
            name, x, y, z = values
            places.append({"name": name, "x": x, "y": y, "z": z})

        return {
            "camera_index": self.camera_index_spin.value(),
            "show_ids": self.show_ids_check.isChecked(),
            "show_advanced": self.settings_toggle.isChecked(),
            "roi_width": self.roi_width_spin.value(),
            "roi_height": self.roi_height_spin.value(),
            "id_bl": self.id_bl_spin.value(),
            "id_br": self.id_br_spin.value(),
            "id_tr": self.id_tr_spin.value(),
            "id_tl": self.id_tl_spin.value(),
            "anchor_mode": self.anchor_mode_combo.currentText(),
            "corner_bl": self.corner_bl_combo.currentText(),
            "corner_br": self.corner_br_combo.currentText(),
            "corner_tr": self.corner_tr_combo.currentText(),
            "corner_tl": self.corner_tl_combo.currentText(),
            "world_bl_x": self.world_bl_x.value(),
            "world_bl_y": self.world_bl_y.value(),
            "world_br_x": self.world_br_x.value(),
            "world_br_y": self.world_br_y.value(),
            "world_tr_x": self.world_tr_x.value(),
            "world_tr_y": self.world_tr_y.value(),
            "world_tl_x": self.world_tl_x.value(),
            "world_tl_y": self.world_tl_y.value(),
            "target_x": self.target_x_spin.value(),
            "target_y": self.target_y_spin.value(),
            "target_z": self.target_z_spin.value(),
            "feed": self.feed_spin.value(),
            "gripper_open_dist": self.gripper_open_dist_spin.value(),
            "gripper_close_dist": self.gripper_close_dist_spin.value(),
            "gripper_feed": self.gripper_feed_spin.value(),
            "jog_x_step": self.jog_x_step_spin.value(),
            "jog_y_step": self.jog_y_step_spin.value(),
            "jog_z_step": self.jog_z_step_spin.value(),
            "places": places,
            "pnp": self._pnp_get_config(),
            "detect_model_path": self._detector.model_path,
            "detect_confidence": self.detect_conf_spin.value(),
            "detect_enabled": self.detect_enable_check.isChecked(),
            "detect_require_roi": self.detect_require_roi_check.isChecked(),
            "detect_class_filter": self.detect_class_combo.currentText(),
            "detect_idle_timeout": self.detect_idle_timeout_spin.value(),
            "detect_wake_delay": self.detect_wake_delay_spin.value(),
            "detect_park_sequence": self.detect_park_edit.toPlainText(),
            "detect_wake_sequence": self.detect_wake_edit.toPlainText(),
            "detect_colors": {str(k): list(v) for k, v in self.det_colors.items()},
        }

    def apply_settings(self, settings: dict[str, Any]) -> None:
        self.camera_index_spin.setValue(int(settings.get("camera_index", self.camera_index_spin.value())))
        self.show_ids_check.setChecked(bool(settings.get("show_ids", self.show_ids_check.isChecked())))
        self.settings_toggle.setChecked(bool(settings.get("show_advanced", self.settings_toggle.isChecked())))
        self.roi_width_spin.setValue(int(settings.get("roi_width", self.roi_width_spin.value())))
        self.roi_height_spin.setValue(int(settings.get("roi_height", self.roi_height_spin.value())))

        self.id_bl_spin.setValue(int(settings.get("id_bl", self.id_bl_spin.value())))
        self.id_br_spin.setValue(int(settings.get("id_br", self.id_br_spin.value())))
        self.id_tr_spin.setValue(int(settings.get("id_tr", self.id_tr_spin.value())))
        self.id_tl_spin.setValue(int(settings.get("id_tl", self.id_tl_spin.value())))

        self.anchor_mode_combo.setCurrentText(str(settings.get("anchor_mode", self.anchor_mode_combo.currentText())))
        self.corner_bl_combo.setCurrentText(str(settings.get("corner_bl", self.corner_bl_combo.currentText())))
        self.corner_br_combo.setCurrentText(str(settings.get("corner_br", self.corner_br_combo.currentText())))
        self.corner_tr_combo.setCurrentText(str(settings.get("corner_tr", self.corner_tr_combo.currentText())))
        self.corner_tl_combo.setCurrentText(str(settings.get("corner_tl", self.corner_tl_combo.currentText())))

        self.world_bl_x.setValue(float(settings.get("world_bl_x", self.world_bl_x.value())))
        self.world_bl_y.setValue(float(settings.get("world_bl_y", self.world_bl_y.value())))
        self.world_br_x.setValue(float(settings.get("world_br_x", self.world_br_x.value())))
        self.world_br_y.setValue(float(settings.get("world_br_y", self.world_br_y.value())))
        self.world_tr_x.setValue(float(settings.get("world_tr_x", self.world_tr_x.value())))
        self.world_tr_y.setValue(float(settings.get("world_tr_y", self.world_tr_y.value())))
        self.world_tl_x.setValue(float(settings.get("world_tl_x", self.world_tl_x.value())))
        self.world_tl_y.setValue(float(settings.get("world_tl_y", self.world_tl_y.value())))

        self.target_x_spin.setValue(float(settings.get("target_x", self.target_x_spin.value())))
        self.target_y_spin.setValue(float(settings.get("target_y", self.target_y_spin.value())))
        self.target_z_spin.setValue(float(settings.get("target_z", self.target_z_spin.value())))
        self.feed_spin.setValue(float(settings.get("feed", self.feed_spin.value())))
        # gripper settings (open/close distances + feed)
        self.gripper_open_dist_spin.setValue(float(settings.get("gripper_open_dist", self.gripper_open_dist_spin.value())))
        self.gripper_close_dist_spin.setValue(float(settings.get("gripper_close_dist", self.gripper_close_dist_spin.value())))
        self.gripper_feed_spin.setValue(float(settings.get("gripper_feed", self.gripper_feed_spin.value())))
        self.jog_x_step_spin.setValue(float(settings.get("jog_x_step", settings.get("jog_xy_step", self.jog_x_step_spin.value()))))
        self.jog_y_step_spin.setValue(float(settings.get("jog_y_step", settings.get("jog_xy_step", self.jog_y_step_spin.value()))))
        self.jog_z_step_spin.setValue(float(settings.get("jog_z_step", self.jog_z_step_spin.value())))

        self.place_table.setRowCount(0)
        for place in settings.get("places", []):
            row = self.place_table.rowCount()
            self.place_table.insertRow(row)
            self._set_place_row(
                row,
                str(place.get("name", f"P{row + 1}")),
                float(place.get("x", 0.0)),
                float(place.get("y", 0.0)),
                float(place.get("z", 0.0)),
            )

        pnp = settings.get("pnp", {})
        if pnp:
            self._pnp_apply_config(pnp)
        self._pnp_sync_place_combo()

        # Detection settings
        model_path = settings.get("detect_model_path", "")
        if model_path and CandyDetector.is_available():
            try:
                self._detector.load(model_path)
                name = Path(model_path).stem
                n_cls = len(self._detector.class_names)
                self.detect_model_label.setText(f"{name} · {n_cls} classes")
                self.detect_enable_check.setEnabled(True)
                self.detect_auto_pick_button.setEnabled(True)
                self.detect_class_combo.clear()
                self.detect_class_combo.addItem("All Classes")
                for cls_name in self._detector.class_names:
                    self.detect_class_combo.addItem(cls_name)
            except Exception:
                pass  # model file may no longer exist

        self.detect_conf_spin.setValue(float(settings.get("detect_confidence", 0.5)))
        self.detect_enable_check.setChecked(bool(settings.get("detect_enabled", False)))
        self.detect_require_roi_check.setChecked(bool(settings.get("detect_require_roi", True)))
        cls_filter = settings.get("detect_class_filter", "All Classes")
        idx = self.detect_class_combo.findText(str(cls_filter))
        if idx >= 0:
            self.detect_class_combo.setCurrentIndex(idx)

        self.detect_idle_timeout_spin.setValue(float(settings.get("detect_idle_timeout", 1.0)))
        self.detect_wake_delay_spin.setValue(float(settings.get("detect_wake_delay", 0.5)))
        park_seq = settings.get("detect_park_sequence", "")
        if park_seq:
            self.detect_park_edit.setPlainText(park_seq)
        wake_seq = settings.get("detect_wake_sequence", "")
        if wake_seq:
            self.detect_wake_edit.setPlainText(wake_seq)

        saved_colors = settings.get("detect_colors", {})
        for k, v in saved_colors.items():
            try:
                self.det_colors[int(k)] = (int(v[0]), int(v[1]), int(v[2]))
            except (ValueError, TypeError, IndexError):
                pass
