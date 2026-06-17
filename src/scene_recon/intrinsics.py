from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2

from scene_recon.record import Record


@dataclass(frozen=True)
class CameraIntrinsics:
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
        return (
            f"unknown unknown {self.width} {self.height} brown {self.focal_x:.4f}"
        )

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


def camera_intrinsics_for_record(record: Record) -> CameraIntrinsics:
    fx, fy, cx, cy, k1, k2, p1, p2, k3 = load_intrinsic_k(record.intrinsics)
    width, height = video_frame_size(record.video)
    return CameraIntrinsics(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        k1=k1,
        k2=k2,
        p1=p1,
        p2=p2,
        k3=k3,
        width=width,
        height=height,
    )


def cameras_json_for_record(record: Record) -> dict[str, dict[str, object]]:
    intrinsics = camera_intrinsics_for_record(record)
    return {intrinsics.camera_id(): intrinsics.to_odm_entry()}


def write_cameras_json(record: Record, output_path: Path) -> CameraIntrinsics:
    intrinsics = camera_intrinsics_for_record(record)
    payload = {intrinsics.camera_id(): intrinsics.to_odm_entry()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    return intrinsics
