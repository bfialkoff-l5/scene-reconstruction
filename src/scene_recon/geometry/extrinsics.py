from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CameraPose:
    easting: float
    northing: float
    alt_m: float
    roll_rad: float
    pitch_rad: float
    yaw_rad: float

    @classmethod
    def from_row(cls, row: pd.Series) -> "CameraPose":
        return cls(
            easting=float(row["easting"]),
            northing=float(row["northing"]),
            alt_m=float(row["altamsl"]),
            roll_rad=float(row.get("roll_rad", 0.0) or 0.0),
            pitch_rad=float(row.get("pitch_rad", 0.0) or 0.0),
            yaw_rad=float(row.get("yaw_rad", 0.0) or 0.0),
        )

    def origin_enu(self) -> np.ndarray:
        return np.array([self.easting, self.northing, self.alt_m], dtype=float)

    def R_cam_to_enu(self) -> np.ndarray:
        nadir = np.pi / 2 + self.pitch_rad
        s = np.sin(nadir)
        f = np.array(
            [np.sin(self.yaw_rad) * s, np.cos(self.yaw_rad) * s, -np.cos(nadir)],
            dtype=float,
        )
        f = f / np.linalg.norm(f)

        up = np.array([0.0, 0.0, 1.0])
        right0 = np.cross(f, up)
        n = np.linalg.norm(right0)
        if n < 1e-8:
            # near-nadir degeneracy: f ~ +/-up; use yaw azimuth as the fallback.
            azimuth = np.array([np.sin(self.yaw_rad), np.cos(self.yaw_rad), 0.0])
            right0 = np.cross(f, azimuth)
            n = np.linalg.norm(right0)
        right0 = right0 / n
        down0 = np.cross(f, right0)
        down0 = down0 / np.linalg.norm(down0)

        c, sr = np.cos(self.roll_rad), np.sin(self.roll_rad)
        right = c * right0 + sr * down0
        down = -sr * right0 + c * down0
        return np.column_stack([right, down, f])


def world_rays(camera, pose: CameraPose, pixels: np.ndarray):
    rays_cam = camera.pixels_to_rays(pixels)
    R = pose.R_cam_to_enu()
    dirs = rays_cam @ R.T
    origins = np.broadcast_to(pose.origin_enu(), dirs.shape).copy()
    return origins, dirs
