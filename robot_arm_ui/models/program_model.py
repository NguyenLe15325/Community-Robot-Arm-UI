from __future__ import annotations

from pathlib import Path


class ProgramModel:
    def __init__(self) -> None:
        self._lines: list[str] = []

    @property
    def lines(self) -> list[str]:
        return list(self._lines)

    def set_lines(self, lines: list[str]) -> None:
        self._lines = [line.rstrip() for line in lines]

    def add_line(self, line: str) -> None:
        text = line.strip()
        if text:
            self._lines.append(text)

    def clear(self) -> None:
        self._lines.clear()

    def save(self, file_path: str) -> None:
        path = Path(file_path)
        path.write_text("\n".join(self._lines) + "\n", encoding="utf-8")

    def load(self, file_path: str) -> list[str]:
        path = Path(file_path)
        loaded = [line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()]
        self._lines = loaded
        return self.lines

    def executable_lines(self) -> list[str]:
        result: list[str] = []
        for line in self._lines:
            stripped = line.strip()
            if not stripped or stripped.startswith(";"):
                continue
            result.append(stripped)
        return result
