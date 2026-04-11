from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from PyQt6.QtCore import QObject, pyqtSignal

from .serial_client import SerialClient


@dataclass
class QueuedCommand:
    text: str
    source: str = "manual"


class CommandExecutor(QObject):
    queue_changed = pyqtSignal(int)
    command_dispatched = pyqtSignal(str, str)
    command_completed = pyqtSignal(str, str, bool, str)
    status = pyqtSignal(str)

    def __init__(self, serial_client: SerialClient) -> None:
        super().__init__()
        self._serial = serial_client
        self._queue: deque[QueuedCommand] = deque()
        self._waiting_ack = False
        self._current: QueuedCommand | None = None
        self._paused = False

        self._serial.line_received.connect(self.on_serial_line)
        self._serial.connection_changed.connect(self._on_connection_changed)

    @property
    def waiting_ack(self) -> bool:
        return self._waiting_ack

    @property
    def queue_size(self) -> int:
        return len(self._queue)

    def enqueue(self, command: str, source: str = "manual") -> None:
        text = command.strip()
        if not text:
            return
        self._queue.append(QueuedCommand(text=text, source=source))
        self.queue_changed.emit(len(self._queue))
        self._try_send_next()

    def enqueue_many(self, commands: list[str], source: str = "program") -> None:
        for command in commands:
            text = command.strip()
            if text:
                self._queue.append(QueuedCommand(text=text, source=source))
        self.queue_changed.emit(len(self._queue))
        self._try_send_next()

    def clear_pending(self) -> None:
        self._queue.clear()
        self.queue_changed.emit(0)

    def set_paused(self, paused: bool) -> None:
        self._paused = paused
        self.status.emit("Queue paused" if paused else "Queue resumed")
        if not paused:
            self._try_send_next()

    def on_serial_line(self, line: str) -> None:
        lower = line.strip().lower()
        if lower == "ok":
            if self._waiting_ack and self._current is not None:
                sent = self._current
                self._current = None
                self._waiting_ack = False
                self.command_completed.emit(sent.text, sent.source, True, line)
                self._try_send_next()
            return

        if lower.startswith("error:") or lower.startswith("alarm:"):
            if self._waiting_ack and self._current is not None:
                sent = self._current
                self._current = None
                self._waiting_ack = False
                self.command_completed.emit(sent.text, sent.source, False, line)
                self._try_send_next()

    def _try_send_next(self) -> None:
        if self._paused or self._waiting_ack or not self._serial.is_connected:
            return
        if not self._queue:
            return

        self._current = self._queue.popleft()
        self.queue_changed.emit(len(self._queue))

        assert self._current is not None
        sent = self._serial.send_line(self._current.text)
        if not sent:
            self.command_completed.emit(self._current.text, self._current.source, False, "send failed")
            self._current = None
            self._waiting_ack = False
            self._try_send_next()
            return

        self._waiting_ack = True
        self.command_dispatched.emit(self._current.text, self._current.source)

    def _on_connection_changed(self, connected: bool) -> None:
        if not connected:
            self._waiting_ack = False
            self._current = None
