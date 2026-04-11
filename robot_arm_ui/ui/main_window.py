from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt
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

        self._build_ui()
        self._connect_signals()
        self._refresh_ports()
        self._apply_styles()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.serial.disconnect_port()
        super().closeEvent(event)

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)

        wrapper = QVBoxLayout(root)
        wrapper.setContentsMargins(16, 16, 16, 16)
        wrapper.setSpacing(12)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        wrapper.addWidget(self.tabs)

        self.tabs.addTab(self._wrap_scroll(self._build_control_page()), "Control")
        self.tabs.addTab(self._wrap_scroll(self._build_program_page()), "Program")
        self.tabs.addTab(self._wrap_scroll(self._build_terminal_page()), "Terminal")
        self.tabs.addTab(self._wrap_scroll(self._build_status_page()), "Status")

    def _wrap_scroll(self, content: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QScrollArea.Shape.NoFrame)
        area.setWidget(content)
        return area

    def _build_control_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(14)
        layout.addWidget(self._build_connection_group())
        layout.addWidget(self._build_frame_group())
        layout.addWidget(self._build_move_group())
        layout.addWidget(self._build_jog_group())
        layout.addWidget(self._build_joint_group())
        layout.addWidget(self._build_quick_group())
        layout.addStretch(1)
        return page

    def _build_program_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(14)
        layout.addWidget(self._build_program_group(), stretch=1)
        return page

    def _build_terminal_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(14)
        layout.addWidget(self._build_terminal_group(), stretch=1)
        return page

    def _build_status_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(14)
        layout.addWidget(self._build_status_group())
        layout.addStretch(1)
        return page

    def _build_connection_group(self) -> QGroupBox:
        group = QGroupBox("Connection")
        layout = QGridLayout(group)
        group.setMinimumWidth(520)

        self.port_combo = QComboBox()
        self.refresh_button = QPushButton("Refresh")

        self.baud_combo = QComboBox()
        for baud in (9600, 57600, 115200, 230400):
            self.baud_combo.addItem(str(baud))
        self.baud_combo.setCurrentText("115200")

        self.connect_button = QPushButton("Connect")
        self.connection_label = QLabel("Disconnected")
        self.connection_label.setObjectName("ConnectionStatus")

        self.port_combo.setMinimumWidth(180)
        self.baud_combo.setMinimumWidth(120)
        self.connect_button.setMinimumWidth(110)

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
        group = QGroupBox("World Frame Transform (robot -> world)")
        form = QFormLayout(group)
        group.setMinimumWidth(520)

        self.tx_spin = self._float_spin(-1000.0, 1000.0, 0.0)
        self.ty_spin = self._float_spin(-1000.0, 1000.0, 0.0)
        self.tz_spin = self._float_spin(-1000.0, 1000.0, 0.0)
        self.roll_spin = self._float_spin(-180.0, 180.0, 0.0)
        self.pitch_spin = self._float_spin(-180.0, 180.0, 0.0)
        self.yaw_spin = self._float_spin(-180.0, 180.0, 0.0)

        self.apply_frame_button = QPushButton("Apply Transform")

        form.addRow("Tx (mm)", self.tx_spin)
        form.addRow("Ty (mm)", self.ty_spin)
        form.addRow("Tz (mm)", self.tz_spin)
        form.addRow("Roll (deg)", self.roll_spin)
        form.addRow("Pitch (deg)", self.pitch_spin)
        form.addRow("Yaw (deg)", self.yaw_spin)
        form.addRow(self.apply_frame_button)

        return group

    def _build_move_group(self) -> QGroupBox:
        group = QGroupBox("Cartesian Move")
        layout = QGridLayout(group)
        group.setMinimumWidth(520)

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
        group.setMinimumWidth(520)

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
        group.setMinimumWidth(520)

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
        layout = QHBoxLayout(group)
        group.setMinimumWidth(520)

        self.cmd_m17 = QPushButton("M17")
        self.cmd_m18 = QPushButton("M18")
        self.cmd_g28 = QPushButton("G28")
        self.cmd_m114 = QPushButton("M114")
        self.cmd_m119 = QPushButton("M119")
        self.cmd_m112 = QPushButton("M112")

        for button in (self.cmd_m17, self.cmd_m18, self.cmd_g28, self.cmd_m114, self.cmd_m119, self.cmd_m112):
            layout.addWidget(button)

        return group

    def _build_status_group(self) -> QGroupBox:
        group = QGroupBox("Status")
        layout = QFormLayout(group)
        group.setMinimumWidth(680)

        self.robot_pose_label = QLabel("X: --  Y: --  Z: --")
        self.world_pose_label = QLabel("Xw: --  Yw: --  Zw: --")
        self.queue_label = QLabel("Queue: 0")

        layout.addRow("Robot Pose", self.robot_pose_label)
        layout.addRow("World Pose", self.world_pose_label)
        layout.addRow("Execution", self.queue_label)

        return group

    def _build_terminal_group(self) -> QGroupBox:
        group = QGroupBox("Terminal")
        layout = QVBoxLayout(group)
        group.setMinimumWidth(680)

        self.terminal_output = QPlainTextEdit()
        self.terminal_output.setReadOnly(True)
        self.terminal_output.setMinimumHeight(420)

        command_row = QHBoxLayout()
        self.terminal_input = QLineEdit()
        self.terminal_input.setPlaceholderText("Type G-code command (example: G1 X100 Y80 Z0 F40)")
        self.send_terminal_button = QPushButton("Send")

        command_row.addWidget(self.terminal_input, stretch=1)
        command_row.addWidget(self.send_terminal_button)

        layout.addWidget(self.terminal_output, stretch=1)
        layout.addLayout(command_row)

        return group

    def _build_program_group(self) -> QGroupBox:
        group = QGroupBox("Program")
        layout = QVBoxLayout(group)
        group.setMinimumWidth(680)

        self.record_checkbox = QCheckBox("Record manual commands into program")
        self.program_editor = QPlainTextEdit()
        self.program_editor.setPlaceholderText("One G-code command per line. ';' starts a comment line.")
        self.program_editor.setMinimumHeight(500)

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

        layout.addWidget(self.record_checkbox)
        layout.addWidget(self.program_editor, stretch=1)
        layout.addLayout(button_row)

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

        self.send_terminal_button.clicked.connect(self._send_terminal_line)
        self.terminal_input.returnPressed.connect(self._send_terminal_line)

        self.program_load_button.clicked.connect(self._load_program)
        self.program_save_button.clicked.connect(self._save_program)
        self.program_clear_button.clicked.connect(self.program_editor.clear)
        self.program_run_button.clicked.connect(self._run_program)
        self.program_stop_button.clicked.connect(self._stop_program)

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
        spin.setMinimumWidth(130)
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
        self._log("World transform updated")

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

    def _submit_command(self, command: str, source: str, recordable: bool) -> None:
        text = command.strip()
        if not text:
            return

        if recordable and source == "manual" and self.record_checkbox.isChecked():
            self.program_editor.appendPlainText(text)

        self.executor.enqueue(text, source=source)

    def _submit_commands(self, commands: list[str], source: str, recordable: bool) -> None:
        cmd_list = [c.strip() for c in commands if c.strip()]
        if not cmd_list:
            return

        if recordable and source == "manual" and self.record_checkbox.isChecked():
            for cmd in cmd_list:
                self.program_editor.appendPlainText(cmd)

        self.executor.enqueue_many(cmd_list, source=source)

    def _send_terminal_line(self) -> None:
        command = self.terminal_input.text().strip()
        if not command:
            return
        self._submit_command(command, source="manual", recordable=True)
        self.terminal_input.clear()

    def _load_program(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Program",
            str(Path.cwd() / "programs"),
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
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Program",
            str(Path.cwd() / "programs" / "program.gcode"),
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
        commands = self.program_model.executable_lines()
        if not commands:
            QMessageBox.information(self, "Program", "No executable commands found.")
            return

        self.executor.enqueue_many(commands, source="program")
        self._log(f"Program queued: {len(commands)} command(s)")

    def _stop_program(self) -> None:
        self.executor.clear_pending()
        self._emergency_stop()

    def _emergency_stop(self) -> None:
        self.executor.clear_pending()
        self.serial.send_line("M112")
        self._log("Emergency stop sent")

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

    def _on_queue_changed(self, size: int) -> None:
        self._queue_size = size
        state = "waiting ack" if self.executor.waiting_ack else "idle"
        self.queue_label.setText(f"Queue: {size} ({state})")

    def _on_command_dispatched(self, command: str, source: str) -> None:
        self._log(f"TX [{source}]: {command}")
        self._on_queue_changed(self._queue_size)

    def _on_command_completed(self, command: str, source: str, success: bool, response: str) -> None:
        flag = "OK" if success else "FAIL"
        self._log(f"DONE [{source}] {flag}: {command} | {response}")
        self._on_queue_changed(self._queue_size)

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
                font-size: 13px;
            }
            QGroupBox {
                border: 1px solid #2d3740;
                border-radius: 10px;
                margin-top: 14px;
                padding: 8px;
                background: #161c22;
                font-weight: 600;
                color: #f0f4f7;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 5px;
                color: #c7d2da;
            }
            QPushButton {
                background: #2f81f7;
                color: #f8fbff;
                border: 0;
                border-radius: 8px;
                padding: 9px 12px;
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
                border-radius: 6px;
                padding: 6px;
            }
            QTabWidget::pane {
                border: 1px solid #2d3740;
                top: -1px;
                background: #11161b;
            }
            QTabBar::tab {
                background: #1a2229;
                color: #b9c6cf;
                padding: 10px 18px;
                border: 1px solid #2d3740;
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                min-width: 100px;
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
