from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSpinBox,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QCheckBox,
    QApplication,
    QSlider,
)

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from robot_arm_ui.core.command_executor import CommandExecutor
from robot_arm_ui.core.frame_transform import FrameTransform
from robot_arm_ui.core.gcode_utils import build_cartesian_move, build_joint_move, fmt_float, parse_m114
from robot_arm_ui.core.serial_client import SerialClient
from robot_arm_ui.models.program_model import ProgramModel
from robot_arm_ui.ui.vision_widget import VisionWidget


class TerminalCommandLineEdit(QLineEdit):
    history_prev_requested = pyqtSignal()
    history_next_requested = pyqtSignal()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # type: ignore[override]
        if event.key() == Qt.Key.Key_Up:
            self.history_prev_requested.emit()
            return
        if event.key() == Qt.Key.Key_Down:
            self.history_next_requested.emit()
            return
        super().keyPressEvent(event)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Community Robot Arm UI")
        self.resize(1400, 900)

        self.serial = SerialClient()
        self.executor = CommandExecutor(self.serial)
        self.program_model = ProgramModel()
        self.transform = FrameTransform()
        self._queue_size = 0
        self._settings_path = self._default_external_settings_path()
        self._terminal_history: list[str] = []
        self._terminal_history_index = 0
        self._terminal_draft = ""

        self._auto_sort_active = False
        self._auto_sort_parked = False
        self._auto_sort_idle_ticks = 0
        self._auto_sort_wake_ticks = 0
        self._teach_pending = False
        self._last_m114: dict[str, float] | None = None
        self._program_running = False
        self._program_loops_remaining = 0
        self._program_commands: list[str] = []
        from PyQt6.QtCore import QTimer
        self._auto_sort_timer = QTimer(self)
        self._auto_sort_timer.setInterval(100)  # poll 10 times a second
        self._auto_sort_timer.timeout.connect(self._on_auto_sort_tick)

        self._build_ui()
        self._connect_signals()
        self._refresh_ports()
        self._load_app_settings(silent=True)
        self._apply_styles()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._save_app_settings(silent=True)
        if hasattr(self, "vision_widget") and self.vision_widget is not None:
            self.vision_widget.stop()
        self.serial.disconnect_port()
        super().closeEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # type: ignore[override]
        if event.key() == Qt.Key.Key_Escape:
            self._emergency_stop()
            return
        super().keyPressEvent(event)

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        # --- Left sidebar (always visible) ---
        sidebar = QWidget()
        sidebar.setMaximumWidth(340)
        sb = QVBoxLayout(sidebar)
        sb.setContentsMargins(0, 0, 0, 0)
        sb.setSpacing(6)
        sb.addWidget(self._build_connection_group())
        sb.addWidget(self._build_status_group())
        sb.addWidget(self._build_speed_override_group())
        sb.addWidget(self._build_quick_group())
        sb.addStretch(1)
        outer.addWidget(sidebar)

        # --- Right tabs ---
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(self._wrap_scroll(self._build_control_page()), "Control")
        self.tabs.addTab(self._build_vision_page(), "Vision")
        self.tabs.addTab(self._build_program_page(), "Program")
        self.tabs.addTab(self._build_terminal_page(), "Terminal")
        outer.addWidget(self.tabs, stretch=1)

    def _build_vision_page(self) -> QWidget:
        self.vision_widget = VisionWidget()
        self.vision_widget.move_request_world_xy.connect(self._on_vision_move_request)
        self.vision_widget.gripper_request.connect(self._on_vision_gripper_request)
        self.vision_widget.place_request_world.connect(self._on_vision_place_request)
        self.vision_widget.pick_and_place_request.connect(self._on_pick_and_place_request)
        self.vision_widget.auto_sort_toggled.connect(self._on_auto_sort_toggled)
        self.vision_widget.pnp_stop_button.clicked.connect(self._pnp_stop)
        self.vision_widget.save_settings_requested.connect(self._save_app_settings_as)
        self.vision_widget.load_settings_requested.connect(self._load_app_settings_from)
        return self.vision_widget

    def _default_external_settings_path(self) -> Path:
        docs = Path.home() / "Documents"
        base = docs if docs.exists() else Path.home()
        return base / "Community-Robot-Arm-UI" / "ui_settings.json"

    def _wrap_scroll(self, content: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QScrollArea.Shape.NoFrame)
        area.setWidget(content)
        return area

    def _build_control_page(self) -> QWidget:
        page = QWidget()
        grid = QGridLayout(page)
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setSpacing(8)
        # Left column
        grid.addWidget(self._build_frame_group(), 0, 0)
        grid.addWidget(self._build_move_group(), 1, 0)
        grid.addWidget(self._build_jog_group(), 2, 0)
        # Right column
        grid.addWidget(self._build_joint_group(), 0, 1)
        grid.addWidget(self._build_gripper_group(), 1, 1)
        # Full width bottom
        grid.addWidget(self._build_teach_group(), 3, 0, 1, 2)
        grid.setRowStretch(3, 1)
        return page

    def _build_program_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)
        layout.addWidget(self._build_program_group(), stretch=1)
        return page

    def _build_speed_override_group(self) -> QGroupBox:
        group = QGroupBox("Speed Override")
        layout = QHBoxLayout(group)

        self.speed_override_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_override_slider.setRange(1, 100)
        self.speed_override_slider.setValue(100)
        self.speed_override_slider.setTickInterval(10)
        self.speed_override_slider.setTickPosition(QSlider.TickPosition.TicksBelow)

        self.speed_override_label = QLabel("100%")
        self.speed_override_label.setMinimumWidth(50)
        self.speed_override_slider.valueChanged.connect(
            lambda v: self.speed_override_label.setText(f"{v}%")
        )

        layout.addWidget(QLabel("Speed"))
        layout.addWidget(self.speed_override_slider, stretch=1)
        layout.addWidget(self.speed_override_label)
        return group

    def _build_teach_group(self) -> QGroupBox:
        from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView

        group = QGroupBox("Teach Points")
        layout = QVBoxLayout(group)

        self.teach_table = QTableWidget(0, 10)
        self.teach_table.setHorizontalHeaderLabels(
            ["Name", "Xr", "Yr", "Zr", "Xw", "Yw", "Zw", "θ1", "θ2", "θ3"]
        )
        header = self.teach_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.teach_table.setAlternatingRowColors(True)
        self.teach_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.teach_table.setMinimumHeight(330)
        layout.addWidget(self.teach_table, stretch=1)

        btn_row = QHBoxLayout()
        self.teach_point_button = QPushButton("Teach Current Pos")
        self.teach_point_button.setStyleSheet("background: #2d8f5a; color: #ffffff; font-weight: bold;")
        self.teach_goto_button = QPushButton("Go To Selected")
        self.teach_insert_button = QPushButton("Insert into Program")
        self.teach_delete_button = QPushButton("Delete")
        self.teach_delete_button.setStyleSheet("background: #b34a4a; color: #ffffff;")

        btn_row.addWidget(self.teach_point_button)
        btn_row.addWidget(self.teach_goto_button)
        btn_row.addWidget(self.teach_insert_button)
        btn_row.addWidget(self.teach_delete_button)
        layout.addLayout(btn_row)

        return group

    def _build_terminal_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)
        layout.addWidget(self._build_terminal_group(), stretch=1)
        return page

    def _build_connection_group(self) -> QGroupBox:
        group = QGroupBox("Connection")
        layout = QGridLayout(group)

        self.port_combo = QComboBox()
        self.refresh_button = QPushButton("Refresh")

        self.baud_combo = QComboBox()
        for baud in (9600, 57600, 115200, 230400):
            self.baud_combo.addItem(str(baud))
        self.baud_combo.setCurrentText("115200")

        self.connect_button = QPushButton("Connect")
        self.connection_label = QLabel("Disconnected")
        self.connection_label.setObjectName("ConnectionStatus")

        self.port_combo.setMinimumWidth(100)
        self.baud_combo.setMinimumWidth(100)
        self.connect_button.setMinimumWidth(90)

        layout.addWidget(QLabel("Port"), 0, 0)
        layout.addWidget(self.port_combo, 0, 1)
        layout.addWidget(self.refresh_button, 0, 2)
        layout.addWidget(QLabel("Baud"), 1, 0)
        layout.addWidget(self.baud_combo, 1, 1)
        layout.addWidget(self.connect_button, 1, 2)
        layout.addWidget(QLabel("Status"), 2, 0)
        layout.addWidget(self.connection_label, 2, 1, 1, 2)

        return group

    def _build_frame_group(self) -> QGroupBox:
        group = QGroupBox("World Frame Transform")
        form = QFormLayout(group)

        self.tx_spin = self._float_spin(-1000.0, 1000.0, 0.0)
        self.ty_spin = self._float_spin(-1000.0, 1000.0, 0.0)
        self.tz_spin = self._float_spin(-1000.0, 1000.0, 0.0)
        self.roll_spin = self._float_spin(-180.0, 180.0, 0.0)
        self.pitch_spin = self._float_spin(-180.0, 180.0, 0.0)
        self.yaw_spin = self._float_spin(-180.0, 180.0, 0.0)

        self.tx_spin.setToolTip("World->robot translation offset on X (mm).")
        self.ty_spin.setToolTip("World->robot translation offset on Y (mm).")
        self.tz_spin.setToolTip("World->robot translation offset on Z (mm).")
        self.roll_spin.setToolTip("Rotation about X axis in degrees.")
        self.pitch_spin.setToolTip("Rotation about Y axis in degrees.")
        self.yaw_spin.setToolTip("Rotation about Z axis in degrees.")

        form.addRow("Tx (mm)", self.tx_spin)
        form.addRow("Ty (mm)", self.ty_spin)
        form.addRow("Tz (mm)", self.tz_spin)
        form.addRow("Roll (deg)", self.roll_spin)
        form.addRow("Pitch (deg)", self.pitch_spin)
        form.addRow("Yaw (deg)", self.yaw_spin)

        self.apply_frame_button = QPushButton("Apply Transform")
        self.apply_frame_button.setToolTip(
            "World→Robot: p_robot = R * (p_world - t)\n"
            "Robot→World: p_world = R^T * p_robot + t"
        )
        form.addRow(self.apply_frame_button)

        return group

    def _build_move_group(self) -> QGroupBox:
        group = QGroupBox("Cartesian Move")
        layout = QGridLayout(group)

        self.frame_combo = QComboBox()
        self.frame_combo.addItems(["Robot", "World"])

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Absolute", "Relative"])

        self.feed_spin = self._float_spin(1.0, 180.0, 40.0)

        self.move_x = self._float_spin(-400.0, 400.0, 100.0)
        self.move_y = self._float_spin(-400.0, 400.0, 80.0)
        self.move_z = self._float_spin(-400.0, 400.0, 0.0)

        self.send_move_button = QPushButton("Send Move")

        layout.addWidget(QLabel("Frame"), 0, 0)
        layout.addWidget(self.frame_combo, 0, 1)
        layout.addWidget(QLabel("Mode"), 0, 2)
        layout.addWidget(self.mode_combo, 0, 3)
        layout.addWidget(QLabel("Feed (deg/s)"), 1, 0)
        layout.addWidget(self.feed_spin, 1, 1)
        layout.addWidget(QLabel("X"), 2, 0)
        layout.addWidget(self.move_x, 2, 1)
        layout.addWidget(QLabel("Y"), 2, 2)
        layout.addWidget(self.move_y, 2, 3)
        layout.addWidget(QLabel("Z"), 3, 0)
        layout.addWidget(self.move_z, 3, 1)
        layout.addWidget(self.send_move_button, 3, 2, 1, 2)

        return group

    def _build_jog_group(self) -> QGroupBox:
        group = QGroupBox("Jog")
        layout = QGridLayout(group)

        self.jog_step_spin = self._float_spin(0.1, 100.0, 5.0)
        self.jog_step_spin.setSingleStep(0.5)
        self.jog_feed_spin = self._float_spin(1.0, 180.0, 30.0)

        self.jog_x_minus = QPushButton("X-")
        self.jog_x_plus = QPushButton("X+")
        self.jog_y_minus = QPushButton("Y-")
        self.jog_y_plus = QPushButton("Y+")
        self.jog_z_minus = QPushButton("Z-")
        self.jog_z_plus = QPushButton("Z+")

        layout.addWidget(QLabel("Step (mm)"), 0, 0)
        layout.addWidget(self.jog_step_spin, 0, 1)
        layout.addWidget(QLabel("Feed (deg/s)"), 0, 2)
        layout.addWidget(self.jog_feed_spin, 0, 3)

        layout.addWidget(self.jog_x_minus, 1, 0)
        layout.addWidget(self.jog_x_plus, 1, 1)
        layout.addWidget(self.jog_y_minus, 1, 2)
        layout.addWidget(self.jog_y_plus, 1, 3)
        layout.addWidget(self.jog_z_minus, 2, 0)
        layout.addWidget(self.jog_z_plus, 2, 1)

        return group

    def _build_joint_group(self) -> QGroupBox:
        group = QGroupBox("Joint Move")
        layout = QGridLayout(group)

        self.t1_spin = self._float_spin(-90.0, 90.0, 0.0)
        self.t2_spin = self._float_spin(0.0, 130.0, 90.0)
        self.t3_spin = self._float_spin(-17.0, 90.0, 0.0)
        self.joint_feed_spin = self._float_spin(1.0, 180.0, 40.0)

        self.send_joint_button = QPushButton("Send Joint Move")

        layout.addWidget(QLabel("T1 (deg)"), 0, 0)
        layout.addWidget(self.t1_spin, 0, 1)
        layout.addWidget(QLabel("T2 (deg)"), 0, 2)
        layout.addWidget(self.t2_spin, 0, 3)
        layout.addWidget(QLabel("T3 (deg)"), 1, 0)
        layout.addWidget(self.t3_spin, 1, 1)
        layout.addWidget(QLabel("Feed (deg/s)"), 1, 2)
        layout.addWidget(self.joint_feed_spin, 1, 3)
        layout.addWidget(self.send_joint_button, 2, 0, 1, 4)

        return group

    def _build_quick_group(self) -> QGroupBox:
        group = QGroupBox("Quick Commands")
        layout = QGridLayout(group)

        self.cmd_m17 = QPushButton("M17")
        self.cmd_m18 = QPushButton("M18")
        self.cmd_g28 = QPushButton("G28")
        self.cmd_m114 = QPushButton("M114")
        self.cmd_m119 = QPushButton("M119")
        self.cmd_m112 = QPushButton("M112")

        self.cmd_m112.setStyleSheet("background: #d32f2f; color: #fff; font-weight: bold;")
        layout.addWidget(self.cmd_m17, 0, 0)
        layout.addWidget(self.cmd_m18, 0, 1)
        layout.addWidget(self.cmd_g28, 0, 2)
        layout.addWidget(self.cmd_m114, 1, 0)
        layout.addWidget(self.cmd_m119, 1, 1)
        layout.addWidget(self.cmd_m112, 1, 2)

        return group

    def _build_gripper_group(self) -> QGroupBox:
        group = QGroupBox("Gripper")
        layout = QGridLayout(group)

        self.gripper_close_dist_spin = self._float_spin(0.1, 100.0, 5.0)
        self.gripper_open_dist_spin = self._float_spin(0.1, 100.0, 20.0)
        self.gripper_feed_spin = self._float_spin(0.1, 200.0, 4.0)

        self.gripper_close_button = QPushButton("Close (M3)")
        self.gripper_open_button = QPushButton("Open (M5)")
        self.gripper_home_button = QPushButton("Home (M6)")
        self.gripper_status_button = QPushButton("Status (M3001)")

        layout.addWidget(QLabel("Close (mm)"), 0, 0)
        layout.addWidget(self.gripper_close_dist_spin, 0, 1)
        layout.addWidget(QLabel("Open (mm)"), 0, 2)
        layout.addWidget(self.gripper_open_dist_spin, 0, 3)
        layout.addWidget(QLabel("Speed (mm/s)"), 1, 0)
        layout.addWidget(self.gripper_feed_spin, 1, 1)

        layout.addWidget(self.gripper_close_button, 2, 0)
        layout.addWidget(self.gripper_open_button, 2, 1)
        layout.addWidget(self.gripper_home_button, 2, 2)
        layout.addWidget(self.gripper_status_button, 2, 3)

        return group

    def _build_status_group(self) -> QGroupBox:
        group = QGroupBox("Status")
        layout = QFormLayout(group)

        self.robot_pose_label = QLabel("X: --  Y: --  Z: --")
        self.world_pose_label = QLabel("Xw: --  Yw: --  Zw: --")
        self.queue_label = QLabel("Queue: 0")

        layout.addRow("Robot Pose", self.robot_pose_label)
        layout.addRow("World Pose", self.world_pose_label)
        layout.addRow("Execution", self.queue_label)

        self.program_loop_label = QLabel("—")
        layout.addRow("Program", self.program_loop_label)

        btn_row = QHBoxLayout()
        self.status_refresh_button = QPushButton("Refresh (M114)")
        self.status_clear_queue_button = QPushButton("Clear Queue")
        self.status_clear_queue_button.setStyleSheet("background: #b34a4a; color: #ffffff;")
        btn_row.addWidget(self.status_refresh_button)
        btn_row.addWidget(self.status_clear_queue_button)
        btn_row.addStretch(1)
        layout.addRow(btn_row)

        return group

    def _build_terminal_group(self) -> QGroupBox:
        group = QGroupBox("Terminal")
        layout = QVBoxLayout(group)

        self.terminal_output = QPlainTextEdit()
        self.terminal_output.setReadOnly(True)
        self.terminal_output.setMinimumHeight(420)

        command_row = QHBoxLayout()
        self.terminal_input = TerminalCommandLineEdit()
        self.terminal_input.setPlaceholderText("Type G-code command (example: G1 X100 Y80 Z0 F40)")
        self.send_terminal_button = QPushButton("Send")
        self.clear_terminal_button = QPushButton("Clear")

        command_row.addWidget(self.terminal_input, stretch=1)
        command_row.addWidget(self.send_terminal_button)
        command_row.addWidget(self.clear_terminal_button)

        layout.addWidget(self.terminal_output, stretch=1)
        layout.addLayout(command_row)

        return group

    def _build_program_group(self) -> QGroupBox:
        group = QGroupBox("Program")
        layout = QVBoxLayout(group)

        self.record_checkbox = QCheckBox("Record manual commands into program")
        self.program_editor = QPlainTextEdit()
        self.program_editor.setPlaceholderText(
            "One G-code command per line.  ';' starts a comment.\n\n"
            "Subroutines (external .gcode files):\n"
            "  CALL path/to/file.gcode        ; run once\n"
            "  CALL path/to/file.gcode 5      ; run 5 times\n"
            "  CALL path/to/file.gcode 0      ; run forever (until Stop)\n\n"
            "Preamble (runs once before loops):\n"
            "  G28               ; home first\n"
            "  ---               ; split marker\n"
            "  CALL routine.gcode 0   ; loop body\n"
        )
        self.program_editor.setMinimumHeight(300)

        button_row = QHBoxLayout()
        self.program_load_button = QPushButton("Load")
        self.program_save_button = QPushButton("Save")
        self.program_clear_button = QPushButton("Clear")
        self.program_run_button = QPushButton("Execute")
        self.program_stop_button = QPushButton("Stop")

        for button in (
            self.program_load_button,
            self.program_save_button,
            self.program_clear_button,
            self.program_run_button,
            self.program_stop_button,
        ):
            button_row.addWidget(button)

        # Loop count
        loop_row = QHBoxLayout()
        loop_row.addWidget(QLabel("Repeat"))
        self.program_loop_spin = QSpinBox()
        self.program_loop_spin.setRange(0, 9999)
        self.program_loop_spin.setValue(1)
        self.program_loop_spin.setSpecialValueText("∞ (until stop)")
        self.program_loop_spin.setToolTip("Number of times to repeat the program. 0 = run forever until Stop.")
        self.program_loop_spin.setMinimumWidth(140)
        loop_row.addWidget(self.program_loop_spin)
        loop_row.addWidget(QLabel("time(s)"))
        loop_row.addStretch(1)

        layout.addWidget(self.record_checkbox)
        layout.addWidget(self.program_editor, stretch=1)
        layout.addLayout(button_row)
        layout.addLayout(loop_row)

        return group

    def _connect_signals(self) -> None:
        self.refresh_button.clicked.connect(self._refresh_ports)
        self.connect_button.clicked.connect(self._toggle_connection)

        self.apply_frame_button.clicked.connect(self._apply_transform)
        self.send_move_button.clicked.connect(self._send_cartesian_move)
        self.send_joint_button.clicked.connect(self._send_joint_move)

        self.jog_x_minus.clicked.connect(lambda: self._send_jog("x", -1.0))
        self.jog_x_plus.clicked.connect(lambda: self._send_jog("x", 1.0))
        self.jog_y_minus.clicked.connect(lambda: self._send_jog("y", -1.0))
        self.jog_y_plus.clicked.connect(lambda: self._send_jog("y", 1.0))
        self.jog_z_minus.clicked.connect(lambda: self._send_jog("z", -1.0))
        self.jog_z_plus.clicked.connect(lambda: self._send_jog("z", 1.0))

        self.cmd_m17.clicked.connect(lambda: self._submit_command("M17", source="manual", recordable=True))
        self.cmd_m18.clicked.connect(lambda: self._submit_command("M18", source="manual", recordable=True))
        self.cmd_g28.clicked.connect(lambda: self._submit_command("G28", source="manual", recordable=True))
        self.cmd_m114.clicked.connect(lambda: self._submit_command("M114", source="manual", recordable=False))
        self.cmd_m119.clicked.connect(lambda: self._submit_command("M119", source="manual", recordable=False))
        self.cmd_m112.clicked.connect(self._emergency_stop)

        # Gripper controls
        self.gripper_close_button.clicked.connect(lambda: self._send_gripper("close"))
        self.gripper_open_button.clicked.connect(lambda: self._send_gripper("open"))
        self.gripper_home_button.clicked.connect(lambda: self._send_gripper("home"))
        self.gripper_status_button.clicked.connect(lambda: self._send_gripper("status"))

        self.send_terminal_button.clicked.connect(self._send_terminal_line)
        self.terminal_input.returnPressed.connect(self._send_terminal_line)
        self.terminal_input.history_prev_requested.connect(self._terminal_history_prev)
        self.terminal_input.history_next_requested.connect(self._terminal_history_next)
        self.clear_terminal_button.clicked.connect(self.terminal_output.clear)

        self.program_load_button.clicked.connect(self._load_program)
        self.program_save_button.clicked.connect(self._save_program)
        self.program_clear_button.clicked.connect(self.program_editor.clear)
        self.program_run_button.clicked.connect(self._run_program)

        # Teach
        self.teach_point_button.clicked.connect(self._teach_current_pos)
        self.teach_goto_button.clicked.connect(self._teach_goto)
        self.teach_insert_button.clicked.connect(self._teach_insert)
        self.teach_delete_button.clicked.connect(self._teach_delete)
        self.program_stop_button.clicked.connect(self._stop_program)

        # Status
        self.status_refresh_button.clicked.connect(
            lambda: self._submit_command("M114", source="manual", recordable=False)
        )
        self.status_clear_queue_button.clicked.connect(self._clear_queue)

        self.serial.connection_changed.connect(self._on_connection_changed)
        self.serial.error.connect(self._on_serial_error)
        self.serial.line_received.connect(self._on_serial_line)

        self.executor.queue_changed.connect(self._on_queue_changed)
        self.executor.command_dispatched.connect(self._on_command_dispatched)
        self.executor.command_completed.connect(self._on_command_completed)
        self.executor.status.connect(self._log)

    def _float_spin(self, minimum: float, maximum: float, value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setDecimals(3)
        spin.setSingleStep(0.5)
        spin.setMinimumWidth(90)
        return spin

    def _refresh_ports(self) -> None:
        current = self.port_combo.currentText()
        self.port_combo.clear()
        ports = SerialClient.available_ports()
        self.port_combo.addItems(ports)
        if current and current in ports:
            self.port_combo.setCurrentText(current)

    def _toggle_connection(self) -> None:
        if self.serial.is_connected:
            self.serial.disconnect_port()
            return

        port = self.port_combo.currentText().strip()
        if not port:
            QMessageBox.warning(self, "Connection", "Please select a serial port.")
            return

        baud = int(self.baud_combo.currentText())
        self.serial.connect_port(port, baud)

    def _on_connection_changed(self, connected: bool) -> None:
        self.connection_label.setText("Connected" if connected else "Disconnected")
        self.connect_button.setText("Disconnect" if connected else "Connect")
        self._log(f"Connection changed: {'connected' if connected else 'disconnected'}")

    def _on_serial_error(self, message: str) -> None:
        self._log(f"Serial error: {message}")

    def _apply_transform(self) -> None:
        self.transform = FrameTransform(
            tx=self.tx_spin.value(),
            ty=self.ty_spin.value(),
            tz=self.tz_spin.value(),
            roll_deg=self.roll_spin.value(),
            pitch_deg=self.pitch_spin.value(),
            yaw_deg=self.yaw_spin.value(),
        )
        self._log("World->robot transform updated")

    def _save_app_settings(self, silent: bool, target_path: Path | None = None) -> None:
        try:
            path = target_path or self._settings_path
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "transform": {
                    "tx": self.tx_spin.value(),
                    "ty": self.ty_spin.value(),
                    "tz": self.tz_spin.value(),
                    "roll": self.roll_spin.value(),
                    "pitch": self.pitch_spin.value(),
                    "yaw": self.yaw_spin.value(),
                },
                "vision": self.vision_widget.get_settings() if hasattr(self, "vision_widget") else {},
                "teach_points": self._get_teach_points(),
            }
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self._settings_path = path
            if not silent:
                self._log(f"Settings saved: {path}")
        except Exception as exc:
            if not silent:
                QMessageBox.warning(self, "Settings", f"Failed to save settings:\n{exc}")

    def _load_app_settings(self, silent: bool, source_path: Path | None = None) -> None:
        path = source_path or self._settings_path
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))

            transform = payload.get("transform", {})
            self.tx_spin.setValue(float(transform.get("tx", self.tx_spin.value())))
            self.ty_spin.setValue(float(transform.get("ty", self.ty_spin.value())))
            self.tz_spin.setValue(float(transform.get("tz", self.tz_spin.value())))
            self.roll_spin.setValue(float(transform.get("roll", self.roll_spin.value())))
            self.pitch_spin.setValue(float(transform.get("pitch", self.pitch_spin.value())))
            self.yaw_spin.setValue(float(transform.get("yaw", self.yaw_spin.value())))
            self._apply_transform()

            vision = payload.get("vision", {})
            if hasattr(self, "vision_widget") and isinstance(vision, dict):
                self.vision_widget.apply_settings(vision)

                # Sync control-page gripper settings from saved vision settings
                try:
                    self.gripper_close_dist_spin.setValue(float(vision.get("gripper_close_dist", self.gripper_close_dist_spin.value())))
                    self.gripper_open_dist_spin.setValue(float(vision.get("gripper_open_dist", self.gripper_open_dist_spin.value())))
                    self.gripper_feed_spin.setValue(float(vision.get("gripper_feed", self.gripper_feed_spin.value())))
                except Exception:
                    pass

            # Teach points
            teach_pts = payload.get("teach_points", [])
            self._set_teach_points(teach_pts)

            self._settings_path = path

            if not silent:
                self._log(f"Settings loaded: {path}")
        except Exception as exc:
            if not silent:
                QMessageBox.warning(self, "Settings", f"Failed to load settings:\n{exc}")

    def _save_app_settings_as(self) -> None:
        config_dir = Path("config")
        config_dir.mkdir(parents=True, exist_ok=True)
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Save Settings File",
            str(config_dir / "ui_settings.json"),
            "JSON Files (*.json);;All Files (*.*)",
        )
        if not path_str:
            return
        self._save_app_settings(silent=False, target_path=Path(path_str))

    def _load_app_settings_from(self) -> None:
        config_dir = Path("config")
        config_dir.mkdir(parents=True, exist_ok=True)
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Load Settings File",
            str(config_dir),
            "JSON Files (*.json);;All Files (*.*)",
        )
        if not path_str:
            return
        self._load_app_settings(silent=False, source_path=Path(path_str))

    def _send_cartesian_move(self) -> None:
        frame = self.frame_combo.currentText().lower()
        mode = self.mode_combo.currentText().lower()
        feed = self.feed_spin.value()

        x = self.move_x.value()
        y = self.move_y.value()
        z = self.move_z.value()

        if mode == "absolute":
            if frame == "world":
                x, y, z = self.transform.world_to_robot_position(x, y, z)
            commands = ["G90", build_cartesian_move(x, y, z, feed)]
        else:
            if frame == "world":
                x, y, z = self.transform.world_delta_to_robot_delta(x, y, z)
            commands = ["G91", build_cartesian_move(x, y, z, feed)]

        self._submit_commands(commands, source="manual", recordable=True)

    def _send_joint_move(self) -> None:
        command = build_joint_move(
            t1=self.t1_spin.value(),
            t2=self.t2_spin.value(),
            t3=self.t3_spin.value(),
            feedrate=self.joint_feed_spin.value(),
        )
        self._submit_command(command, source="manual", recordable=True)

    def _send_jog(self, axis: str, direction: float) -> None:
        step = self.jog_step_spin.value() * direction
        feed = self.jog_feed_spin.value()

        dx = dy = dz = 0.0
        if axis == "x":
            dx = step
        elif axis == "y":
            dy = step
        elif axis == "z":
            dz = step

        if self.frame_combo.currentText().lower() == "world":
            dx, dy, dz = self.transform.world_delta_to_robot_delta(dx, dy, dz)

        commands = [
            "G91",
            build_cartesian_move(dx, dy, dz, feed),
            "G90",
        ]
        self._submit_commands(commands, source="manual", recordable=True)

    def _apply_speed_override(self, command: str) -> str:
        """Scale F (feedrate) values by the speed override percentage."""
        import re
        override = self.speed_override_slider.value() / 100.0
        if override >= 1.0:
            return command

        def scale_f(m: re.Match) -> str:
            original = float(m.group(1))
            scaled = original * override
            return f"F{fmt_float(scaled)}"

        return re.sub(r'F([+-]?\d*\.?\d+)', scale_f, command, flags=re.IGNORECASE)

    def _submit_command(self, command: str, source: str, recordable: bool) -> None:
        text = command.strip()
        if not text:
            return

        if recordable and source == "manual" and self.record_checkbox.isChecked():
            self.program_editor.appendPlainText(text)

        text = self._apply_speed_override(text)
        self.executor.enqueue(text, source=source)

    def _submit_commands(self, commands: list[str], source: str, recordable: bool) -> None:
        cmd_list = [c.strip() for c in commands if c.strip()]
        if not cmd_list:
            return

        if recordable and source == "manual" and self.record_checkbox.isChecked():
            for cmd in cmd_list:
                self.program_editor.appendPlainText(cmd)

        cmd_list = [self._apply_speed_override(c) for c in cmd_list]
        self.executor.enqueue_many(cmd_list, source=source)

    def _send_terminal_line(self) -> None:
        command = self.terminal_input.text().strip()
        if not command:
            return

        if not self._terminal_history or self._terminal_history[-1] != command:
            self._terminal_history.append(command)
        self._terminal_history_index = len(self._terminal_history)
        self._terminal_draft = ""

        self._submit_command(command, source="manual", recordable=True)
        self.terminal_input.clear()

    def _terminal_history_prev(self) -> None:
        if not self._terminal_history:
            return

        if self._terminal_history_index == len(self._terminal_history):
            self._terminal_draft = self.terminal_input.text()

        if self._terminal_history_index > 0:
            self._terminal_history_index -= 1

        self.terminal_input.setText(self._terminal_history[self._terminal_history_index])

    def _terminal_history_next(self) -> None:
        if not self._terminal_history:
            return

        if self._terminal_history_index < len(self._terminal_history) - 1:
            self._terminal_history_index += 1
            self.terminal_input.setText(self._terminal_history[self._terminal_history_index])
        else:
            self._terminal_history_index = len(self._terminal_history)
            self.terminal_input.setText(self._terminal_draft)

    def _send_gripper(self, action: str) -> None:
        if action == "close":
            dist = self.gripper_close_dist_spin.value()
        elif action == "open":
            dist = self.gripper_open_dist_spin.value()
        else:
            dist = 0.0

        feed = self.gripper_feed_spin.value()

        if action == "close":
            # M3 S<mm> F<mm/s>
            cmd = f"M3 S{fmt_float(dist)} F{fmt_float(feed)}"
            self._submit_command(cmd, source="manual", recordable=True)
        elif action == "open":
            cmd = f"M5 S{fmt_float(dist)} F{fmt_float(feed)}"
            self._submit_command(cmd, source="manual", recordable=True)
        elif action == "home":
            cmd = f"M6 F{fmt_float(feed)}"
            self._submit_command(cmd, source="manual", recordable=True)
        elif action == "status":
            cmd = "M3001"
            self._submit_command(cmd, source="manual", recordable=False)

    def _on_vision_move_request(self, wx: float, wy: float, wz: float, feed: float) -> None:
        rx, ry, rz = self.transform.world_to_robot_position(wx, wy, wz)
        commands = ["G90", build_cartesian_move(rx, ry, rz, feed)]
        self._submit_commands(commands, source="manual", recordable=True)
        self._log(f"Vision target queued: world X={fmt_float(wx)} Y={fmt_float(wy)} Z={fmt_float(wz)}")

    def _on_vision_gripper_request(self, action: str, dist: float, feed: float) -> None:
        if action == "close":
            cmd = f"M3 S{fmt_float(dist)} F{fmt_float(feed)}"
            self._submit_command(cmd, source="manual", recordable=True)
        elif action == "open":
            cmd = f"M5 S{fmt_float(dist)} F{fmt_float(feed)}"
            self._submit_command(cmd, source="manual", recordable=True)
        elif action == "home":
            cmd = f"M6 F{fmt_float(feed)}"
            self._submit_command(cmd, source="manual", recordable=True)
        elif action == "status":
            self._submit_command("M3001", source="manual", recordable=False)

    def _on_vision_place_request(
        self,
        wx: float,
        wy: float,
        wz: float,
        feed: float,
        grip_action: str,
        grip_dist: float,
        grip_feed: float,
    ) -> None:
        rx, ry, rz = self.transform.world_to_robot_position(wx, wy, wz)
        commands = ["G90", build_cartesian_move(rx, ry, rz, feed)]

        if grip_action == "open":
            commands.append(f"M5 S{fmt_float(grip_dist)} F{fmt_float(grip_feed)}")
        elif grip_action == "close":
            commands.append(f"M3 S{fmt_float(grip_dist)} F{fmt_float(grip_feed)}")

        self._submit_commands(commands, source="manual", recordable=True)
        self._log(
            f"Vision place queued: world X={fmt_float(wx)} Y={fmt_float(wy)} Z={fmt_float(wz)} grip={grip_action}"
        )

    def _on_pick_and_place_request(self, gcode_lines: list) -> None:
        """Convert world-coordinate G-code to robot coordinates and enqueue.

        Lines starting with G0 or G1 that contain X/Y/Z parameters are
        treated as world-coordinate moves: the XYZ values are converted
        through the vision transform before dispatch.  All other lines
        (M3, M5, G4, G90, etc.) are passed through unchanged.

        The firmware now sends 'ok' only after motion completes, so no
        M400 workaround is needed.
        """
        import re
        commands: list[str] = []

        for line in gcode_lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Check if this is a G0/G1 move with world-coordinate XYZ
            m = re.match(r'^G[01]\b', stripped, re.IGNORECASE)
            if m:
                # Extract X, Y, Z, F parameters
                x_m = re.search(r'X([+-]?\d*\.?\d+)', stripped, re.IGNORECASE)
                y_m = re.search(r'Y([+-]?\d*\.?\d+)', stripped, re.IGNORECASE)
                z_m = re.search(r'Z([+-]?\d*\.?\d+)', stripped, re.IGNORECASE)
                f_m = re.search(r'F([+-]?\d*\.?\d+)', stripped, re.IGNORECASE)

                if x_m or y_m or z_m:
                    wx = float(x_m.group(1)) if x_m else 0.0
                    wy = float(y_m.group(1)) if y_m else 0.0
                    wz = float(z_m.group(1)) if z_m else 0.0
                    feed = float(f_m.group(1)) if f_m else 30.0

                    rx, ry, rz = self.transform.world_to_robot_position(wx, wy, wz)
                    commands.append(build_cartesian_move(rx, ry, rz, feed))
                    continue

            # Pass through unchanged (M3, M5, G4, G90, etc.)
            commands.append(stripped)

        if not commands:
            self._log("Pick & Place: nothing to execute")
            return

        # Update UI state
        self.vision_widget.pnp_execute_button.setEnabled(False)
        self.vision_widget.pnp_stop_button.setEnabled(True)

        self._submit_commands(commands, source="pnp", recordable=True)
        self._log(f"Pick & Place: {len(commands)} command(s) queued")

    def _pnp_stop(self) -> None:
        """Cancel the running PnP sequence."""
        self.executor.clear_pending()
        self._auto_sort_active = False
        self._auto_sort_parked = False
        self._auto_sort_idle_ticks = 0
        self._auto_sort_wake_ticks = 0
        self._auto_sort_timer.stop()
        self.vision_widget.detect_auto_pick_button.setChecked(False)
        self.vision_widget._on_auto_sort_clicked(False)
        self._log("Pick & Place: stopped")
        self.vision_widget.pnp_execute_button.setEnabled(True)
        self.vision_widget.pnp_stop_button.setEnabled(False)
        self.vision_widget.status_label.setText("Pick & Place: stopped")

    def _on_auto_sort_toggled(self, active: bool) -> None:
        """Handle start/stop of continuous auto-sort."""
        self._auto_sort_active = active
        if active:
            if not self.serial.is_connected:
                self._log("Auto Sort: Error - Robot not connected")
                self.vision_widget.detect_auto_pick_button.setChecked(False)
                self.vision_widget._on_auto_sort_clicked(False)
                return
            self._log("Auto Sort: Started continuous sorting")
            self._auto_sort_timer.start()
            self.vision_widget.pnp_execute_button.setEnabled(False)
            self.vision_widget.pnp_stop_button.setEnabled(True)
        else:
            self._auto_sort_timer.stop()
            self._log("Auto Sort: Stopped")
            self.vision_widget.pnp_execute_button.setEnabled(True)
            self.vision_widget.pnp_stop_button.setEnabled(False)

    def _on_auto_sort_tick(self) -> None:
        """Polling loop for Look-Pick-Look state machine.

        States:
          - active, not parked: picking candies one-by-one
          - active, parked: motors disabled, waiting for detection
          - idle ticks accumulating: no detection for consecutive ticks
        """
        if not self._auto_sort_active:
            return

        # Ensure robot is still connected
        if not self.serial.is_connected:
            self.vision_widget.detect_auto_pick_button.setChecked(False)
            self.vision_widget._on_auto_sort_clicked(False)
            self._log("Auto Sort: Stopped - Robot disconnected")
            return

        # Wait until robot is completely idle
        if self.executor.queue_size > 0 or self.executor.waiting_ack:
            return

        # Robot is idle! Get the current best detection
        best_det = self.vision_widget.get_best_detection()

        if best_det:
            self._auto_sort_idle_ticks = 0  # reset idle counter
            candy_name, wx, wy = best_det

            if self._auto_sort_parked:
                # Require sustained detection before waking
                self._auto_sort_wake_ticks += 1
                wake_delay_s = self.vision_widget.detect_wake_delay_spin.value()
                wake_threshold = max(1, int(wake_delay_s / 0.1))

                remaining = wake_threshold - self._auto_sort_wake_ticks
                if remaining > 0:
                    self.vision_widget.detect_status_label.setText(
                        f"Parked — confirming detection ({remaining} ticks)"
                    )
                    return  # keep waiting for sustained detection

                # Sustained detection confirmed — wake up
                self._auto_sort_parked = False
                self._auto_sort_wake_ticks = 0
                self._log("Auto Sort: Sustained detection confirmed — waking up")
                self.vision_widget.detect_status_label.setText("Waking up...")
                wake_cmds = self._parse_gcode_lines(
                    self.vision_widget.detect_wake_edit.toPlainText()
                )
                if wake_cmds:
                    self._submit_commands(wake_cmds, source="pnp", recordable=True)
                # The pick will happen on the *next* tick after wake completes
            else:
                # Normal pick
                self._generate_and_queue_single_pick(candy_name, wx, wy)
        else:
            # No detection right now
            self._auto_sort_wake_ticks = 0  # reset wake counter
            if self._auto_sort_parked:
                return  # already parked, just wait

            self._auto_sort_idle_ticks += 1

            # Compute tick threshold from user-defined timeout (seconds / 0.1s per tick)
            timeout_s = self.vision_widget.detect_idle_timeout_spin.value()
            threshold = max(1, int(timeout_s / 0.1))

            if self._auto_sort_idle_ticks >= threshold:
                self._auto_sort_parked = True
                self._log("Auto Sort: No candies detected — parking")
                self.vision_widget.detect_status_label.setText("Parked (waiting)")
                park_cmds = self._parse_gcode_lines(
                    self.vision_widget.detect_park_edit.toPlainText()
                )
                if park_cmds:
                    self._submit_commands(park_cmds, source="pnp", recordable=True)

    @staticmethod
    def _parse_gcode_lines(text: str) -> list[str]:
        """Parse a G-code text block: strip comments and blank lines."""
        result: list[str] = []
        for line in text.splitlines():
            cmd = line.split(";")[0].strip()
            if cmd:
                result.append(cmd)
        return result

    def _generate_and_queue_single_pick(self, candy_name: str, wx: float, wy: float) -> None:
        """Generate and queue the PnP template for a single candy.

        Each candy's world XY overrides {PICK_X} and {PICK_Y}.
        If a Place exists with the same name as the candy class,
        its coordinates override {PLACE_X}, {PLACE_Y}, and {PLACE_Z}.
        All other variables come from the current UI state.
        """
        import re

        template_text = self.vision_widget.pnp_gcode_edit.toPlainText().strip()
        if not template_text:
            self._log("Auto Pick All: no G-code template")
            return

        base_variables = self.vision_widget._pnp_build_variables()
        commands: list[str] = []

        # Build a lookup of places by lowercased name for auto-sorting
        places_by_name = {}
        for row in range(self.vision_widget.place_table.rowCount()):
            vals = self.vision_widget._row_values(row)
            if vals:
                name, px, py, pz = vals
                places_by_name[name.lower()] = (px, py, pz)

        variables = dict(base_variables)
        # Override pick coordinates for this candy
        variables["PICK_X"] = f"{wx:g}"
        variables["PICK_Y"] = f"{wy:g}"

        # If a place is named exactly after the class, auto-sort it there!
        class_lower = candy_name.lower()
        if class_lower in places_by_name:
            px, py, pz = places_by_name[class_lower]
            variables["PLACE_X"] = f"{px:g}"
            variables["PLACE_Y"] = f"{py:g}"
            variables["PLACE_Z"] = f"{pz:g}"

        for raw_line in template_text.splitlines():
            cmd = raw_line.split(";")[0].strip()
            if not cmd:
                continue
            try:
                resolved = cmd.format_map(variables)
            except (KeyError, ValueError):
                continue

            # Convert world→robot for G0/G1
            m = re.match(r'^G[01]\b', resolved, re.IGNORECASE)
            if m:
                x_m = re.search(r'X([+-]?\d*\.?\d+)', resolved, re.IGNORECASE)
                y_m = re.search(r'Y([+-]?\d*\.?\d+)', resolved, re.IGNORECASE)
                z_m = re.search(r'Z([+-]?\d*\.?\d+)', resolved, re.IGNORECASE)
                f_m = re.search(r'F([+-]?\d*\.?\d+)', resolved, re.IGNORECASE)
                if x_m or y_m or z_m:
                    wx2 = float(x_m.group(1)) if x_m else 0.0
                    wy2 = float(y_m.group(1)) if y_m else 0.0
                    wz2 = float(z_m.group(1)) if z_m else 0.0
                    feed = float(f_m.group(1)) if f_m else 30.0
                    rx, ry, rz = self.transform.world_to_robot_position(wx2, wy2, wz2)
                    commands.append(build_cartesian_move(rx, ry, rz, feed))
                    continue

            commands.append(resolved)

        if not commands:
            self._log("Auto Sort: Failed to generate commands")
            return

        self._submit_commands(commands, source="pnp", recordable=True)
        self._log(f"Auto Sort: Picking {candy_name} at X={wx:.1f}, Y={wy:.1f}")

    def _get_app_data_dir(self, subfolder: str = "") -> Path:
        docs = Path.home() / "Documents"
        base = docs if docs.exists() else Path.home()
        path = base / "Community-Robot-Arm-UI"
        if subfolder:
            path = path / subfolder
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _load_program(self) -> None:
        programs_dir = self._get_app_data_dir("programs")
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Program",
            str(programs_dir),
            "G-code Files (*.gcode *.nc *.txt);;All Files (*.*)",
        )
        if not path:
            return

        try:
            lines = self.program_model.load(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load Program", f"Failed to load file:\n{exc}")
            return

        self.program_editor.setPlainText("\n".join(lines))
        self._log(f"Program loaded: {path}")

    def _save_program(self) -> None:
        programs_dir = self._get_app_data_dir("programs")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Program",
            str(programs_dir / "program.gcode"),
            "G-code Files (*.gcode *.nc *.txt);;All Files (*.*)",
        )
        if not path:
            return

        lines = self.program_editor.toPlainText().splitlines()
        self.program_model.set_lines(lines)
        try:
            self.program_model.save(path)
        except Exception as exc:
            QMessageBox.critical(self, "Save Program", f"Failed to save file:\n{exc}")
            return

        self._log(f"Program saved: {path}")

    def _run_program(self) -> None:
        lines = self.program_editor.toPlainText().splitlines()
        self.program_model.set_lines(lines)
        raw_commands = self.program_model.executable_lines()
        if not raw_commands:
            QMessageBox.information(self, "Program", "No executable commands found.")
            return

        # Expand CALL directives
        try:
            result = self._expand_calls(raw_commands)
        except RuntimeError as exc:
            QMessageBox.critical(self, "Program", str(exc))
            return

        if isinstance(result, tuple):
            # Infinite CALL: (preamble, loop_body)
            preamble, loop_body = result
            self._program_commands = loop_body
            self._program_loops_total = -1  # infinite
            self._program_loops_remaining = -1
            self._program_current_loop = 0
            self._program_running = True
            # Enqueue preamble first, then the loop body will follow
            if preamble:
                preamble_cmds = [self._apply_speed_override(c) for c in preamble]
                self.executor.enqueue_many(preamble_cmds, source="program_preamble")
                self._log(f"Program: {len(preamble)} preamble cmd(s) queued")
            self._program_enqueue_iteration()
        else:
            # Normal program
            commands = result
            loop_count = self.program_loop_spin.value()  # 0 = infinite

            self._program_commands = commands
            self._program_loops_total = loop_count if loop_count > 0 else -1
            self._program_loops_remaining = self._program_loops_total
            self._program_current_loop = 0
            self._program_running = True
            self._program_enqueue_iteration()

    def _program_enqueue_iteration(self) -> None:
        """Enqueue one iteration of the stored program."""
        if not self._program_running:
            return
        self._program_current_loop += 1
        cmds = [self._apply_speed_override(c) for c in self._program_commands]
        self.executor.enqueue_many(cmds, source="program")

        # Update status display
        if self._program_loops_total > 0:
            self.program_loop_label.setText(
                f"▶ Loop {self._program_current_loop} / {self._program_loops_total}"
            )
            self._log(f"Program: loop {self._program_current_loop}/{self._program_loops_total}")
        else:
            self.program_loop_label.setText(
                f"▶ Loop {self._program_current_loop} (∞)"
            )
            self._log(f"Program: loop {self._program_current_loop} (∞)")

    def _program_on_iteration_done(self) -> None:
        """Called when one program iteration completes. Queue next or stop."""
        if not self._program_running:
            return

        if self._program_loops_remaining > 0:
            self._program_loops_remaining -= 1
            if self._program_loops_remaining == 0:
                self._program_running = False
                self.program_loop_label.setText(
                    f"✓ Complete ({self._program_current_loop} loops)"
                )
                self._log("Program: all loops complete")
                return

        # Queue next iteration
        self._program_enqueue_iteration()

    def _stop_program(self) -> None:
        self._program_running = False
        self._program_loops_remaining = 0
        self.program_loop_label.setText("■ Stopped")
        self.executor.clear_pending()
        self._emergency_stop()

    @staticmethod
    def _expand_calls(
        lines: list[str], depth: int = 0, max_depth: int = 10
    ) -> list[str] | tuple[list[str], list[str]]:
        """Expand CALL directives by inlining the referenced .gcode files.

        Syntax:
            CALL path/to/file.gcode        ; runs once
            CALL path/to/file.gcode 5      ; runs 5 times
            CALL path/to/file.gcode 0      ; runs forever (infinite)

        When repeat=0 (infinite) is encountered at the top level,
        returns a tuple (preamble, loop_body) instead of a flat list.
        Nested CALL is supported up to max_depth levels.
        """
        if depth > max_depth:
            raise RuntimeError(f"CALL nesting too deep (>{max_depth}). Possible recursion.")

        result: list[str] = []
        for line in lines:
            if line.upper().startswith("CALL "):
                args = line[5:].strip()
                # Strip inline comment
                if ";" in args:
                    args = args[:args.index(";")].strip()

                # Parse: path [repeat_count]
                parts = args.rsplit(None, 1)  # split from right
                repeat = 1
                if len(parts) == 2 and parts[1].isdigit():
                    file_path = parts[0]
                    repeat = int(parts[1])
                else:
                    file_path = args

                path = Path(file_path)
                if not path.is_file():
                    raise RuntimeError(f"CALL error: file not found: {file_path}")

                sub_lines = [
                    ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
                    if ln.strip() and not ln.strip().startswith(";")
                ]
                # Recursively expand nested CALLs (nested infinite not allowed)
                expanded = MainWindow._expand_calls(sub_lines, depth + 1, max_depth)
                if isinstance(expanded, tuple):
                    raise RuntimeError("CALL with infinite repeat (0) cannot be nested.")

                if repeat == 0 and depth == 0:
                    # Infinite loop: return (preamble, loop_body)
                    return (result, expanded)
                else:
                    repeat = max(1, repeat)
                    for _ in range(repeat):
                        result.extend(expanded)
            else:
                result.append(line)
        return result

    def _emergency_stop(self) -> None:
        self.executor.clear_pending()
        self.serial.send_line("M112")
        self._log("Emergency stop sent")

    def _clear_queue(self) -> None:
        """Clear all pending commands without emergency stop."""
        self._program_running = False
        self._program_loops_remaining = 0
        self.program_loop_label.setText("— (queue cleared)")
        self.executor.clear_pending()
        self._log("Queue cleared")

    def _on_serial_line(self, line: str) -> None:
        self._log(f"RX: {line}")

        parsed = parse_m114(line)
        if parsed is None:
            return

        rx = parsed.get("X", 0.0)
        ry = parsed.get("Y", 0.0)
        rz = parsed.get("Z", 0.0)
        self.robot_pose_label.setText(f"X: {fmt_float(rx)}  Y: {fmt_float(ry)}  Z: {fmt_float(rz)}")

        wx, wy, wz = self.transform.robot_to_world_position(rx, ry, rz)
        self.world_pose_label.setText(f"Xw: {fmt_float(wx)}  Yw: {fmt_float(wy)}  Zw: {fmt_float(wz)}")

        # Store latest M114 values for Teach feature
        self._last_m114 = parsed

        # Auto-capture for Teach if pending
        if self._teach_pending:
            self._teach_pending = False
            self._teach_add_point(parsed)

    # ------------------------------------------------------------------
    # Teach Points
    # ------------------------------------------------------------------

    def _teach_current_pos(self) -> None:
        """Send M114 and automatically capture the position when the response arrives."""
        self._teach_pending = True
        self._submit_command("M114", source="manual", recordable=False)
        self._log("Teach: Querying position...")

    def _teach_add_point(self, parsed: dict[str, float]) -> None:
        """Add a teach point from parsed M114 data."""
        from PyQt6.QtWidgets import QTableWidgetItem
        rx, ry, rz = parsed.get('X', 0.0), parsed.get('Y', 0.0), parsed.get('Z', 0.0)
        wx, wy, wz = self.transform.robot_to_world_position(rx, ry, rz)

        row = self.teach_table.rowCount()
        self.teach_table.insertRow(row)
        name = f"P{row + 1}"
        vals = [name,
                fmt_float(rx), fmt_float(ry), fmt_float(rz),
                fmt_float(wx), fmt_float(wy), fmt_float(wz),
                fmt_float(parsed.get('T1', 0.0)),
                fmt_float(parsed.get('T2', 0.0)),
                fmt_float(parsed.get('T3', 0.0))]
        for col, val in enumerate(vals):
            self.teach_table.setItem(row, col, QTableWidgetItem(val))
        self._log(f"Teach: Saved {name} — Robot({vals[1]},{vals[2]},{vals[3]}) World({vals[4]},{vals[5]},{vals[6]})")

    def _teach_goto(self) -> None:
        """Move to the selected teach point using the selected frame."""
        row = self.teach_table.currentRow()
        if row < 0:
            return
        try:
            frame = self.frame_combo.currentText().lower()
            if frame == "world":
                # Read world coords, convert to robot for sending
                wx = float(self.teach_table.item(row, 4).text())
                wy = float(self.teach_table.item(row, 5).text())
                wz = float(self.teach_table.item(row, 6).text())
                x, y, z = self.transform.world_to_robot_position(wx, wy, wz)
            else:
                # Read robot coords directly
                x = float(self.teach_table.item(row, 1).text())
                y = float(self.teach_table.item(row, 2).text())
                z = float(self.teach_table.item(row, 3).text())
        except (ValueError, AttributeError):
            return
        feed = self.feed_spin.value()
        commands = ["G90", build_cartesian_move(x, y, z, feed)]
        self._submit_commands(commands, source="manual", recordable=False)
        name = self.teach_table.item(row, 0).text()
        self._log(f"Teach: Moving to {name} (frame={frame})")

    def _teach_insert(self) -> None:
        """Insert the selected teach point as raw G-code into the program editor."""
        row = self.teach_table.currentRow()
        if row < 0:
            return
        try:
            rx = float(self.teach_table.item(row, 1).text())
            ry = float(self.teach_table.item(row, 2).text())
            rz = float(self.teach_table.item(row, 3).text())
        except (ValueError, AttributeError):
            return
        feed = self.feed_spin.value()
        name = self.teach_table.item(row, 0).text()
        self.program_editor.appendPlainText(build_cartesian_move(rx, ry, rz, feed))
        self._log(f"Teach: Inserted {name} into Program")

    def _teach_delete(self) -> None:
        """Delete the selected teach point."""
        row = self.teach_table.currentRow()
        if row >= 0:
            self.teach_table.removeRow(row)

    def _get_teach_points(self) -> list[dict]:
        """Serialize teach table to a list of dicts for JSON persistence."""
        from PyQt6.QtWidgets import QTableWidgetItem
        points: list[dict] = []
        headers = ["name", "xr", "yr", "zr", "xw", "yw", "zw", "t1", "t2", "t3"]
        for row in range(self.teach_table.rowCount()):
            pt: dict = {}
            for col, key in enumerate(headers):
                item = self.teach_table.item(row, col)
                pt[key] = item.text() if item else ""
            points.append(pt)
        return points

    def _set_teach_points(self, points: list[dict]) -> None:
        """Restore teach table from a list of dicts."""
        from PyQt6.QtWidgets import QTableWidgetItem
        self.teach_table.setRowCount(0)
        headers = ["name", "xr", "yr", "zr", "xw", "yw", "zw", "t1", "t2", "t3"]
        for pt in points:
            row = self.teach_table.rowCount()
            self.teach_table.insertRow(row)
            for col, key in enumerate(headers):
                self.teach_table.setItem(row, col, QTableWidgetItem(str(pt.get(key, ""))))

    def _on_queue_changed(self, size: int) -> None:
        self._queue_size = size
        state = "waiting ack" if self.executor.waiting_ack else "idle"
        self.queue_label.setText(f"Queue: {size} ({state})")

        # Re-enable PnP execute button when fully idle
        if size == 0 and not self.executor.waiting_ack:
            if hasattr(self, "vision_widget"):
                if not self.vision_widget.pnp_execute_button.isEnabled():
                    self.vision_widget.pnp_execute_button.setEnabled(True)
                    self.vision_widget.pnp_stop_button.setEnabled(False)
                    self.vision_widget.status_label.setText("Pick & Place: sequence complete")

    def _on_command_dispatched(self, command: str, source: str) -> None:
        self._log(f"TX [{source}]: {command}")
        self._on_queue_changed(self._queue_size)

    def _on_command_completed(self, command: str, source: str, success: bool, response: str) -> None:
        flag = "OK" if success else "FAIL"
        self._log(f"DONE [{source}] {flag}: {command} | {response}")
        self._on_queue_changed(self._queue_size)

        # Program loop: when the last program command completes, queue next iteration
        if (self._program_running
                and source == "program"
                and self.executor.queue_size == 0
                and not self.executor.waiting_ack):
            self._program_on_iteration_done()

    def _log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.terminal_output.appendPlainText(f"[{timestamp}] {message}")

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                background: #11161b;
                color: #e7ecef;
                font-family: 'Segoe UI';
                font-size: 9pt;
            }
            QGroupBox {
                border: 1px solid #2d3740;
                border-radius: 8px;
                margin-top: 18px;
                padding: 6px;
                background: #161c22;
                font-weight: 600;
                color: #f0f4f7;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                top: 2px;
                padding: 0 4px;
                color: #c7d2da;
            }
            QGroupBox::indicator {
                width: 13px;
                height: 13px;
            }
            QPushButton {
                background: #2f81f7;
                color: #f8fbff;
                border: 0;
                border-radius: 6px;
                padding: 6px 10px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #4a93ff;
            }
            QPushButton:pressed {
                background: #2166d4;
            }
            QLineEdit, QPlainTextEdit, QComboBox, QDoubleSpinBox, QSpinBox {
                background: #0f1419;
                color: #eef4f7;
                border: 1px solid #33404a;
                border-radius: 5px;
                padding: 4px;
            }
            QTabWidget::pane {
                border: 1px solid #2d3740;
                top: -1px;
                background: #11161b;
            }
            QTabBar::tab {
                background: #1a2229;
                color: #b9c6cf;
                padding: 8px 16px;
                border: 1px solid #2d3740;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                min-width: 80px;
            }
            QTabBar::tab:selected {
                background: #11161b;
                color: #ffffff;
            }
            QTabBar::tab:hover {
                background: #22303a;
            }
            QLabel#ConnectionStatus {
                font-weight: 700;
                color: #6ee7a8;
            }
            """
        )


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
