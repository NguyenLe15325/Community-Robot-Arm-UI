from __future__ import annotations

from typing import Optional


def fmt_float(value: float, precision: int = 3) -> str:
    text = f"{value:.{precision}f}".rstrip("0").rstrip(".")
    return text if text else "0"


def build_cartesian_move(x: Optional[float], y: Optional[float], z: Optional[float], feedrate: float) -> str:
    parts = ["G1"]
    if x is not None:
        parts.append(f"X{fmt_float(x)}")
    if y is not None:
        parts.append(f"Y{fmt_float(y)}")
    if z is not None:
        parts.append(f"Z{fmt_float(z)}")
    parts.append(f"F{fmt_float(feedrate)}")
    return " ".join(parts)


def build_joint_move(t1: Optional[float], t2: Optional[float], t3: Optional[float], feedrate: float) -> str:
    parts = ["G1"]
    if t1 is not None:
        parts.append(f"T1{fmt_float(t1)}")
    if t2 is not None:
        parts.append(f"T2{fmt_float(t2)}")
    if t3 is not None:
        parts.append(f"T3{fmt_float(t3)}")
    parts.append(f"F{fmt_float(feedrate)}")
    return " ".join(parts)


def parse_m114(line: str) -> Optional[dict[str, float]]:
    # Example: X:120.00 Y:10.00 Z:-30.00 T1:5.00 T2:90.00 T3:0.00
    if "X:" not in line or "Y:" not in line or "Z:" not in line:
        return None

    values: dict[str, float] = {}
    for token in line.strip().split():
        if ":" not in token:
            continue
        key, raw = token.split(":", 1)
        try:
            values[key.upper()] = float(raw)
        except ValueError:
            continue

    required = ("X", "Y", "Z")
    if not all(k in values for k in required):
        return None
    return values
