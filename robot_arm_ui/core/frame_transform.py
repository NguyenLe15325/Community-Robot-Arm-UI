from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin


@dataclass
class FrameTransform:
    # Translation and rotation that maps robot frame -> world frame.
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0

    def _rotation_matrix(self) -> list[list[float]]:
        rx = radians(self.roll_deg)
        ry = radians(self.pitch_deg)
        rz = radians(self.yaw_deg)

        cx, sx = cos(rx), sin(rx)
        cy, sy = cos(ry), sin(ry)
        cz, sz = cos(rz), sin(rz)

        # R = Rz(yaw) * Ry(pitch) * Rx(roll)
        return [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ]

    @staticmethod
    def _mat_vec_mul(matrix: list[list[float]], vec: tuple[float, float, float]) -> tuple[float, float, float]:
        x = matrix[0][0] * vec[0] + matrix[0][1] * vec[1] + matrix[0][2] * vec[2]
        y = matrix[1][0] * vec[0] + matrix[1][1] * vec[1] + matrix[1][2] * vec[2]
        z = matrix[2][0] * vec[0] + matrix[2][1] * vec[1] + matrix[2][2] * vec[2]
        return x, y, z

    @staticmethod
    def _transpose(matrix: list[list[float]]) -> list[list[float]]:
        return [
            [matrix[0][0], matrix[1][0], matrix[2][0]],
            [matrix[0][1], matrix[1][1], matrix[2][1]],
            [matrix[0][2], matrix[1][2], matrix[2][2]],
        ]

    def robot_to_world_position(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        rot = self._rotation_matrix()
        rx, ry, rz = self._mat_vec_mul(rot, (x, y, z))
        return rx + self.tx, ry + self.ty, rz + self.tz

    def world_to_robot_position(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        rot = self._rotation_matrix()
        inv = self._transpose(rot)
        shifted = (x - self.tx, y - self.ty, z - self.tz)
        return self._mat_vec_mul(inv, shifted)

    def world_delta_to_robot_delta(self, dx: float, dy: float, dz: float) -> tuple[float, float, float]:
        # Relative moves should not be translated, only rotated.
        rot = self._rotation_matrix()
        inv = self._transpose(rot)
        return self._mat_vec_mul(inv, (dx, dy, dz))
