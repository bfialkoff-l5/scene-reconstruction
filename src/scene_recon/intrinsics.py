from __future__ import annotations

import json
from pathlib import Path

from scene_recon.camera import Camera, load_intrinsic_k, video_frame_size
from scene_recon.record import Record

CameraIntrinsics = Camera


def camera_intrinsics_for_record(record: Record) -> Camera:
    return Camera.from_record(record)


def cameras_json_for_record(record: Record) -> dict[str, dict[str, object]]:
    intrinsics = camera_intrinsics_for_record(record)
    return {intrinsics.camera_id(): intrinsics.to_odm_entry()}


def write_cameras_json(record: Record, output_path: Path) -> Camera:
    intrinsics = camera_intrinsics_for_record(record)
    payload = {intrinsics.camera_id(): intrinsics.to_odm_entry()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    return intrinsics
