from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from scene_recon.record import Record


def load_intrinsic_k(path: Path) -> tuple[float, float, float, float, float, float, float, float, float]:
    rows: list[list[str]] = []
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if not row or all(cell.strip() == "" for cell in row):
                continue
            rows.append(row)

    if len(rows) < 4:
        raise ValueError(f"expected at least 4 rows in {path}, got {len(rows)}")

    fx = float(rows[0][0])
    cx = float(rows[0][2])
    fy = float(rows[1][1])
    cy = float(rows[1][2])
    k1, k2, p1, p2, k3 = (float(value) for value in rows[3][:5])
    return fx, fy, cx, cy, k1, k2, p1, p2, k3


def video_frame_size(video_path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()

    if width <= 0 or height <= 0:
        raise RuntimeError(f"invalid frame size from {video_path}: {width}x{height}")
    return width, height


@dataclass(frozen=True)
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float
    k2: float
    p1: float
    p2: float
    k3: float
    width: int
    height: int

    @classmethod
    def from_csv(cls, path: Path, *, width: int, height: int) -> Camera:
        fx, fy, cx, cy, k1, k2, p1, p2, k3 = load_intrinsic_k(path)
        return cls(
            fx=fx, fy=fy, cx=cx, cy=cy,
            k1=k1, k2=k2, p1=p1, p2=p2, k3=k3,
            width=width, height=height,
        )

    @classmethod
    def from_record(cls, record: Record) -> Camera:
        width, height = video_frame_size(record.video)
        return cls.from_csv(record.intrinsics, width=width, height=height)

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def dist(self) -> np.ndarray:
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float64)

    def sample_grid(self, shape: tuple[int, int]) -> np.ndarray:
        us = np.linspace(0.0, self.width - 1, shape[0])
        vs = np.linspace(0.0, self.height - 1, shape[1])
        uu, vv = np.meshgrid(us, vs, indexing="ij")
        return np.stack([uu.ravel(), vv.ravel()], axis=1).astype(np.float64)

    def pixels_to_rays(self, pixels: np.ndarray) -> np.ndarray:
        pts = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
        normalized = cv2.undistortPoints(pts, self.K, self.dist).reshape(-1, 2)
        rays = np.column_stack([normalized, np.ones(len(normalized))])
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)

    def project(self, points_cam: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_cam, dtype=np.float64).reshape(-1, 1, 3)
        zero = np.zeros(3, dtype=np.float64)
        projected, _ = cv2.projectPoints(pts, zero, zero, self.K, self.dist)
        return projected.reshape(-1, 2)

    @property
    def max_dim(self) -> int:
        return max(self.width, self.height)

    @property
    def focal_x(self) -> float:
        return self.fx / self.max_dim

    @property
    def focal_y(self) -> float:
        return self.fy / self.max_dim

    @property
    def c_x(self) -> float:
        return (self.cx - self.width / 2) / self.max_dim

    @property
    def c_y(self) -> float:
        return (self.cy - self.height / 2) / self.max_dim

    def camera_id(self) -> str:
        # Must BYTE-MATCH the camera key ODM derives from our frames, or its
        # --cameras override silently won't bind and ODM self-calibrates from a
        # default focal (observed: it diverged to a principal point 681 px outside
        # the image and OpenMVS fused 0 points). ODM's opendm/photo.py builds the id
        # as " ".join(["v2", make.strip(), model.strip(), w, h, projection,
        # str(focal_ratio)[:6]]).lower(). Our frames are EXIF-less PNGs, so make and
        # model are empty (hence the doubled spaces) and the focal_ratio prior is
        # 0.85. The real calibration rides in to_odm_entry()'s values, not the key.
        # ponytail: pinned to ODM 3.5.x no-EXIF defaults ("v2", 0.85 prior). If we
        # ever embed EXIF focal in the frames, or bump ODM and the key format moves,
        # regenerate this from opensfm/camera_models.json instead of hardcoding.
        return f"v2   {self.width} {self.height} brown 0.85"

    def to_odm_entry(self) -> dict[str, object]:
        return {
            "projection_type": "brown",
            "width": self.width,
            "height": self.height,
            "focal_x": self.focal_x,
            "focal_y": self.focal_y,
            "c_x": self.c_x,
            "c_y": self.c_y,
            "k1": self.k1,
            "k2": self.k2,
            "p1": self.p1,
            "p2": self.p2,
            "k3": self.k3,
        }
