from __future__ import annotations

import threading
from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal
import serial
from serial.tools import list_ports


class SerialClient(QObject):
    line_received = pyqtSignal(str)
    connection_changed = pyqtSignal(bool)
    error = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._serial: Optional[serial.Serial] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._reader_stop = threading.Event()
        self._write_lock = threading.Lock()

    @property
    def is_connected(self) -> bool:
        return self._serial is not None and self._serial.is_open

    @staticmethod
    def available_ports() -> list[str]:
        return [p.device for p in list_ports.comports()]

    def connect_port(self, port: str, baudrate: int) -> bool:
        self.disconnect_port()
        try:
            self._serial = serial.Serial(port=port, baudrate=baudrate, timeout=0.1)
        except Exception as exc:
            self._serial = None
            self.error.emit(f"Connect failed: {exc}")
            self.connection_changed.emit(False)
            return False

        self._reader_stop.clear()
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()
        self.connection_changed.emit(True)
        return True

    def disconnect_port(self) -> None:
        self._reader_stop.set()
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=0.5)
        self._reader_thread = None

        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
        self._serial = None
        self.connection_changed.emit(False)

    def send_line(self, command: str) -> bool:
        if not self.is_connected:
            self.error.emit("Serial not connected")
            return False
        payload = (command.strip() + "\n").encode("ascii", errors="ignore")
        try:
            with self._write_lock:
                assert self._serial is not None
                self._serial.write(payload)
            return True
        except Exception as exc:
            self.error.emit(f"Send failed: {exc}")
            return False

    def _reader_loop(self) -> None:
        while not self._reader_stop.is_set():
            if self._serial is None:
                break
            try:
                raw = self._serial.readline()
            except Exception as exc:
                self.error.emit(f"Serial read error: {exc}")
                self.disconnect_port()
                return

            if not raw:
                continue

            line = raw.decode("utf-8", errors="replace").strip()
            if line:
                self.line_received.emit(line)
