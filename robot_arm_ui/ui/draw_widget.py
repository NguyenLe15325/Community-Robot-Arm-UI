from __future__ import annotations

from math import hypot, sqrt
from typing import List, Tuple

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

Point2D = Tuple[float, float]


class DrawCanvas(QWidget):
    cursor_moved = pyqtSignal(float, float)
    strokes_changed = pyqtSignal(int, int)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(640, 420)
        self.setMouseTracking(True)
        self._bounds = (0.0, 300.0, 0.0, 300.0)  # xmin, xmax, ymin, ymax
        self._min_step = 2.0
        self._lock_aspect = False
        self._strokes: List[List[Point2D]] = []
        self._current: List[Point2D] | None = None
        self._cursor_world: Point2D | None = None
        self._line_anchor: Point2D | None = None
        self._line_mode = False

    def set_bounds(self, xmin: float, xmax: float, ymin: float, ymax: float) -> None:
        if xmax <= xmin:
            xmax = xmin + 1.0
        if ymax <= ymin:
            ymax = ymin + 1.0
        self._bounds = (xmin, xmax, ymin, ymax)
        self.update()

    def set_min_step(self, min_step: float) -> None:
        self._min_step = max(0.0, float(min_step))

    def set_lock_aspect(self, locked: bool) -> None:
        self._lock_aspect = bool(locked)
        self.update()

    def clear(self) -> None:
        self._strokes.clear()
        self._current = None
        self._line_anchor = None
        self._line_mode = False
        self._emit_stroke_stats()
        self.update()

    def undo(self) -> None:
        if self._strokes:
            self._strokes.pop()
        self._emit_stroke_stats()
        self.update()

    def strokes_world(self) -> List[List[Point2D]]:
        return [list(stroke) for stroke in self._strokes if stroke]

    def simplified_strokes(self, tolerance: float) -> List[List[Point2D]]:
        if tolerance <= 0.0:
            return self.strokes_world()
        simplified: List[List[Point2D]] = []
        for stroke in self._strokes:
            if not stroke:
                continue
            if len(stroke) < 3:
                simplified.append(list(stroke))
                continue
            simplified.append(self._rdp(stroke, tolerance))
        return simplified

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self._current = []
            self._strokes.append(self._current)
            self._line_mode = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            wx, wy = self._canvas_to_world(event.position().x(), event.position().y())
            self._line_anchor = (wx, wy)
            if self._line_mode:
                self._current.append((wx, wy))
                self._current.append((wx, wy))
            else:
                self._add_point(event.position().x(), event.position().y())
            self._emit_stroke_stats()
            self.update()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        pos = event.position()
        wx, wy = self._canvas_to_world(pos.x(), pos.y())
        self._cursor_world = (wx, wy)
        self.cursor_moved.emit(wx, wy)

        if event.buttons() & Qt.MouseButton.LeftButton and self._current is not None:
            if self._line_mode or (event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
                if not self._line_mode:
                    self._line_mode = True
                    if self._current:
                        self._line_anchor = self._current[0]
                    else:
                        self._line_anchor = (wx, wy)
                self._set_line_end(pos.x(), pos.y())
            else:
                self._add_point(pos.x(), pos.y())
            self._emit_stroke_stats()
            self.update()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self._current = None
            self._line_anchor = None
            self._line_mode = False
            self._emit_stroke_stats()
            self.update()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        painter.fillRect(self.rect(), QColor(13, 18, 23))
        border_pen = QPen(QColor(51, 64, 74), 1)
        painter.setPen(border_pen)
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))

        x0, y0, w, h = self._draw_rect()
        painter.drawRect(int(x0), int(y0), max(0, int(w) - 1), max(0, int(h) - 1))

        grid_pen = QPen(QColor(26, 34, 41), 1)
        painter.setPen(grid_pen)
        w = int(w)
        h = int(h)
        step = 80
        for x in range(int(x0) + step, int(x0) + w, step):
            painter.drawLine(x, int(y0), x, int(y0) + h)
        for y in range(int(y0) + step, int(y0) + h, step):
            painter.drawLine(int(x0), y, int(x0) + w, y)

        stroke_pen = QPen(QColor(110, 231, 168), 2)
        painter.setPen(stroke_pen)

        for stroke in self._strokes:
            if len(stroke) < 2:
                if len(stroke) == 1:
                    px, py = self._world_to_canvas(stroke[0][0], stroke[0][1])
                    painter.drawEllipse(px - 2, py - 2, 4, 4)
                continue
            last_px, last_py = self._world_to_canvas(stroke[0][0], stroke[0][1])
            for wx, wy in stroke[1:]:
                px, py = self._world_to_canvas(wx, wy)
                painter.drawLine(last_px, last_py, px, py)
                last_px, last_py = px, py

    def _add_point(self, canvas_x: float, canvas_y: float) -> None:
        if self._current is None:
            return
        wx, wy = self._canvas_to_world(canvas_x, canvas_y)
        if self._current:
            last = self._current[-1]
            if self._min_step > 0.0 and self._distance(wx, wy, last[0], last[1]) < self._min_step:
                return
        self._current.append((wx, wy))

    def _set_line_end(self, canvas_x: float, canvas_y: float) -> None:
        if self._current is None:
            return
        wx, wy = self._canvas_to_world(canvas_x, canvas_y)
        anchor = self._line_anchor
        if anchor is None:
            anchor = (wx, wy)
            self._line_anchor = anchor
        if not self._current:
            self._current.append(anchor)
            self._current.append((wx, wy))
            return
        self._current[:] = [anchor, (wx, wy)]

    def _emit_stroke_stats(self) -> None:
        stroke_count = len([s for s in self._strokes if s])
        point_count = sum(len(s) for s in self._strokes)
        self.strokes_changed.emit(stroke_count, point_count)

    def _canvas_to_world(self, cx: float, cy: float) -> Point2D:
        xmin, xmax, ymin, ymax = self._bounds
        x0, y0, width, height = self._draw_rect()
        width = max(1.0, float(width))
        height = max(1.0, float(height))
        nx = min(1.0, max(0.0, (cx - x0) / width))
        ny = min(1.0, max(0.0, 1.0 - ((cy - y0) / height)))
        wx = xmin + nx * (xmax - xmin)
        wy = ymin + ny * (ymax - ymin)
        return wx, wy

    def _world_to_canvas(self, wx: float, wy: float) -> Tuple[int, int]:
        xmin, xmax, ymin, ymax = self._bounds
        if xmax <= xmin:
            xmax = xmin + 1.0
        if ymax <= ymin:
            ymax = ymin + 1.0
        nx = (wx - xmin) / (xmax - xmin)
        ny = (wy - ymin) / (ymax - ymin)
        nx = min(1.0, max(0.0, nx))
        ny = min(1.0, max(0.0, ny))
        x0, y0, width, height = self._draw_rect()
        px = int(x0 + nx * width)
        py = int(y0 + (1.0 - ny) * height)
        return px, py

    def _draw_rect(self) -> Tuple[float, float, float, float]:
        w = float(self.width())
        h = float(self.height())
        if not self._lock_aspect:
            return 0.0, 0.0, w, h

        xmin, xmax, ymin, ymax = self._bounds
        world_w = max(1.0, xmax - xmin)
        world_h = max(1.0, ymax - ymin)
        world_aspect = world_w / world_h
        widget_aspect = w / h if h > 0 else world_aspect

        if widget_aspect >= world_aspect:
            height = h
            width = height * world_aspect
            x0 = (w - width) / 2.0
            y0 = 0.0
        else:
            width = w
            height = width / world_aspect
            x0 = 0.0
            y0 = (h - height) / 2.0

        return x0, y0, width, height


    @staticmethod
    def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
        return hypot(x2 - x1, y2 - y1)

    @staticmethod
    def _rdp(points: List[Point2D], epsilon: float) -> List[Point2D]:
        if len(points) < 3:
            return list(points)

        start = points[0]
        end = points[-1]
        dmax = 0.0
        index = 0
        for i in range(1, len(points) - 1):
            d = DrawCanvas._perp_distance(points[i], start, end)
            if d > dmax:
                index = i
                dmax = d

        if dmax > epsilon:
            left = DrawCanvas._rdp(points[: index + 1], epsilon)
            right = DrawCanvas._rdp(points[index:], epsilon)
            return left[:-1] + right
        return [start, end]

    @staticmethod
    def _perp_distance(p: Point2D, a: Point2D, b: Point2D) -> float:
        ax, ay = a
        bx, by = b
        px, py = p
        if ax == bx and ay == by:
            return sqrt((px - ax) ** 2 + (py - ay) ** 2)
        num = abs((by - ay) * px - (bx - ax) * py + bx * ay - by * ax)
        den = sqrt((by - ay) ** 2 + (bx - ax) ** 2)
        return num / den


class DrawWidget(QWidget):
    path_request_world = pyqtSignal(list, float, float, float, float)
    export_program_requested = pyqtSignal(list, float, float, float, float)
    save_gcode_requested = pyqtSignal(list, float, float, float, float)

    def __init__(self) -> None:
        super().__init__()
        self._build_ui()
        self._connect_signals()

    def get_settings(self) -> dict:
        return {
            "bounds": {
                "xmin": self.world_x_min.value(),
                "xmax": self.world_x_max.value(),
                "ymin": self.world_y_min.value(),
                "ymax": self.world_y_max.value(),
            },
            "draw_z": self.draw_z_spin.value(),
            "lift_z": self.lift_z_spin.value(),
            "feed": self.feed_spin.value(),
            "travel_feed": self.travel_feed_spin.value(),
            "min_step": self.min_step_spin.value(),
            "simplify_tol": self.simplify_spin.value(),
            "lock_aspect": self.lock_aspect_check.isChecked(),
        }

    def apply_settings(self, settings: dict) -> None:
        bounds = settings.get("bounds", {}) if isinstance(settings, dict) else {}
        try:
            self.world_x_min.setValue(float(bounds.get("xmin", self.world_x_min.value())))
            self.world_x_max.setValue(float(bounds.get("xmax", self.world_x_max.value())))
            self.world_y_min.setValue(float(bounds.get("ymin", self.world_y_min.value())))
            self.world_y_max.setValue(float(bounds.get("ymax", self.world_y_max.value())))
            self.draw_z_spin.setValue(float(settings.get("draw_z", self.draw_z_spin.value())))
            self.lift_z_spin.setValue(float(settings.get("lift_z", self.lift_z_spin.value())))
            self.feed_spin.setValue(float(settings.get("feed", self.feed_spin.value())))
            self.travel_feed_spin.setValue(float(settings.get("travel_feed", self.travel_feed_spin.value())))
            self.min_step_spin.setValue(float(settings.get("min_step", self.min_step_spin.value())))
            self.simplify_spin.setValue(float(settings.get("simplify_tol", self.simplify_spin.value())))
            self.lock_aspect_check.setChecked(bool(settings.get("lock_aspect", self.lock_aspect_check.isChecked())))
        except Exception:
            return
        self._sync_bounds()
        self._sync_aspect()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(10)

        self.canvas = DrawCanvas()
        layout.addWidget(self.canvas, stretch=1)

        side = QVBoxLayout()
        layout.addLayout(side)

        side.addWidget(self._build_monitor_group())
        side.addWidget(self._build_bounds_group())
        side.addWidget(self._build_tool_group())
        side.addWidget(self._build_simplify_group())
        side.addWidget(self._build_action_group())
        side.addStretch(1)

        self._sync_bounds()
        self._sync_min_step()
        self._sync_aspect()

    def _build_monitor_group(self) -> QGroupBox:
        group = QGroupBox("Monitor")
        layout = QFormLayout(group)
        self.cursor_world_label = QLabel("X: --  Y: --")
        self.stats_label = QLabel("Strokes: 0  Points: 0")
        self.status_label = QLabel("Draw: idle")
        self.tip_label = QLabel("Tip: hold Shift for line")
        layout.addRow("Cursor", self.cursor_world_label)
        layout.addRow("Stats", self.stats_label)
        layout.addRow("Status", self.status_label)
        layout.addRow("Tip", self.tip_label)
        return group

    def _build_bounds_group(self) -> QGroupBox:
        group = QGroupBox("World Bounds (mm)")
        layout = QGridLayout(group)

        self.world_x_min = self._float_spin(0.0)
        self.world_x_max = self._float_spin(300.0)
        self.world_y_min = self._float_spin(0.0)
        self.world_y_max = self._float_spin(300.0)
        self.lock_aspect_check = QCheckBox("Lock aspect ratio")
        self.lock_aspect_check.setChecked(False)

        layout.addWidget(QLabel("X min"), 0, 0)
        layout.addWidget(self.world_x_min, 0, 1)
        layout.addWidget(QLabel("X max"), 0, 2)
        layout.addWidget(self.world_x_max, 0, 3)

        layout.addWidget(QLabel("Y min"), 1, 0)
        layout.addWidget(self.world_y_min, 1, 1)
        layout.addWidget(QLabel("Y max"), 1, 2)
        layout.addWidget(self.world_y_max, 1, 3)

        layout.addWidget(self.lock_aspect_check, 2, 0, 1, 4)
        return group


    def _build_tool_group(self) -> QGroupBox:
        group = QGroupBox("Tool")
        layout = QFormLayout(group)

        self.draw_z_spin = self._float_spin(0.0)
        self.lift_z_spin = self._float_spin(20.0)
        self.feed_spin = self._float_spin(30.0)
        self.travel_feed_spin = self._float_spin(60.0)

        layout.addRow("Draw Z", self.draw_z_spin)
        layout.addRow("Lift Z", self.lift_z_spin)
        layout.addRow("Draw Feed", self.feed_spin)
        layout.addRow("Travel Feed", self.travel_feed_spin)
        return group

    def _build_simplify_group(self) -> QGroupBox:
        group = QGroupBox("Simplify")
        layout = QFormLayout(group)

        self.min_step_spin = self._float_spin(2.0)
        self.simplify_spin = self._float_spin(1.0)
        layout.addRow("Min step", self.min_step_spin)
        layout.addRow("Simplify tol", self.simplify_spin)
        return group

    def _build_action_group(self) -> QGroupBox:
        group = QGroupBox("Actions")
        layout = QVBoxLayout(group)

        row1 = QHBoxLayout()
        row2 = QHBoxLayout()

        self.undo_button = QPushButton("Undo")
        self.clear_button = QPushButton("Clear")
        self.queue_button = QPushButton("Queue Path")
        self.queue_button.setStyleSheet("background: #2d8f5a; color: #ffffff;")
        self.export_program_button = QPushButton("Export to Program")
        self.save_gcode_button = QPushButton("Save G-code")

        row1.addWidget(self.undo_button)
        row1.addWidget(self.clear_button)
        row1.addWidget(self.queue_button)

        row2.addWidget(self.export_program_button)
        row2.addWidget(self.save_gcode_button)

        layout.addLayout(row1)
        layout.addLayout(row2)
        return group

    def _connect_signals(self) -> None:
        self.world_x_min.valueChanged.connect(self._sync_bounds)
        self.world_x_max.valueChanged.connect(self._sync_bounds)
        self.world_y_min.valueChanged.connect(self._sync_bounds)
        self.world_y_max.valueChanged.connect(self._sync_bounds)
        self.lock_aspect_check.toggled.connect(self._sync_aspect)
        self.min_step_spin.valueChanged.connect(self._sync_min_step)

        self.canvas.cursor_moved.connect(self._on_cursor_moved)
        self.canvas.strokes_changed.connect(self._on_strokes_changed)

        self.undo_button.clicked.connect(self.canvas.undo)
        self.clear_button.clicked.connect(self.canvas.clear)
        self.queue_button.clicked.connect(self._queue_path)
        self.export_program_button.clicked.connect(self._export_program)
        self.save_gcode_button.clicked.connect(self._save_gcode)


    def _sync_bounds(self) -> None:
        xmin = self.world_x_min.value()
        xmax = self.world_x_max.value()
        ymin = self.world_y_min.value()
        ymax = self.world_y_max.value()
        if xmax <= xmin:
            xmax = xmin + 1.0
            self.world_x_max.setValue(xmax)
        if ymax <= ymin:
            ymax = ymin + 1.0
            self.world_y_max.setValue(ymax)
        self.canvas.set_bounds(xmin, xmax, ymin, ymax)

    def _sync_min_step(self) -> None:
        self.canvas.set_min_step(self.min_step_spin.value())

    def _sync_aspect(self) -> None:
        self.canvas.set_lock_aspect(self.lock_aspect_check.isChecked())

    def _on_cursor_moved(self, wx: float, wy: float) -> None:
        self.cursor_world_label.setText(f"X: {wx:.2f}  Y: {wy:.2f}")

    def _on_strokes_changed(self, strokes: int, points: int) -> None:
        self.stats_label.setText(f"Strokes: {strokes}  Points: {points}")

    def _build_request_payload(self) -> tuple[list, float, float, float, float] | None:
        strokes = self.canvas.simplified_strokes(self.simplify_spin.value())
        if not strokes:
            self.status_label.setText("Draw: no strokes")
            return None

        draw_z = self.draw_z_spin.value()
        lift_z = self.lift_z_spin.value()
        feed = self.feed_spin.value()
        travel_feed = self.travel_feed_spin.value()
        return strokes, draw_z, lift_z, feed, travel_feed

    def _queue_path(self) -> None:
        payload = self._build_request_payload()
        if payload is None:
            return
        strokes, draw_z, lift_z, feed, travel_feed = payload
        self.path_request_world.emit(strokes, draw_z, lift_z, feed, travel_feed)
        self.status_label.setText(f"Draw: queued {sum(len(s) for s in strokes)} pt(s)")

    def _export_program(self) -> None:
        payload = self._build_request_payload()
        if payload is None:
            return
        strokes, draw_z, lift_z, feed, travel_feed = payload
        self.export_program_requested.emit(strokes, draw_z, lift_z, feed, travel_feed)
        self.status_label.setText("Draw: exported to Program")

    def _save_gcode(self) -> None:
        payload = self._build_request_payload()
        if payload is None:
            return
        strokes, draw_z, lift_z, feed, travel_feed = payload
        self.save_gcode_requested.emit(strokes, draw_z, lift_z, feed, travel_feed)
        self.status_label.setText("Draw: save requested")

    @staticmethod
    def _float_spin(value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-10000.0, 10000.0)
        spin.setDecimals(3)
        spin.setSingleStep(1.0)
        spin.setValue(value)
        spin.setMinimumWidth(110)
        return spin
