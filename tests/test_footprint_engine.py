from __future__ import annotations

import numpy as np
from affine import Affine

from scene_recon.camera import Camera
from scene_recon.geometry.extrinsics import CameraPose
from scene_recon.geometry.footprint import ground_footprint
from scene_recon.geometry.terrain import TerrainModel
from scene_recon.selection.grid import GroundGrid


def _flat_terrain(size: int = 400, cell: float = 10.0) -> TerrainModel:
    # world (col,row)->(easting,northing): origin top-left, north decreasing with row.
    transform = Affine.translation(-2000.0, 2000.0) * Affine.scale(cell, -cell)
    array = np.zeros((size, size), dtype=np.float32)
    return TerrainModel.from_array(array, transform, nodata=-9999.0)


def _camera() -> Camera:
    return Camera(
        fx=2274.0, fy=2276.0, cx=960.0, cy=540.0,
        k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0,
        width=1920, height=1080,
    )


def _grid() -> GroundGrid:
    return GroundGrid(bin_size_m=5.0, origin_e=-2000.0, origin_n=-2000.0)


def test_nadir_footprint_is_valid_block() -> None:
    cam = _camera()
    terrain = _flat_terrain()
    grid = _grid()
    # pitch=-pi/2 => optical axis straight down; centered at origin, 150 m up.
    pose = CameraPose(
        easting=0.0, northing=0.0, alt_m=150.0,
        roll_rad=0.0, pitch_rad=-np.pi / 2, yaw_rad=0.0,
    )
    fp = ground_footprint(cam, pose, terrain, grid, frame_number=0, ray_grid=(24, 14))
    assert fp.valid
    assert fp.valid_frac == 1.0
    assert len(fp.cells) > 4
    # footprint centroid sits under the camera for a nadir view
    assert abs(fp.centroid_e) < 30.0
    assert abs(fp.centroid_n) < 30.0


def test_horizon_footprint_invalid() -> None:
    cam = _camera()
    terrain = _flat_terrain()
    grid = _grid()
    # pitch=0 => optical axis horizontal; rays point at/above horizon, no ground hit.
    pose = CameraPose(
        easting=0.0, northing=0.0, alt_m=150.0,
        roll_rad=0.0, pitch_rad=0.0, yaw_rad=0.0,
    )
    fp = ground_footprint(
        cam, pose, terrain, grid, frame_number=1, ray_grid=(24, 14), max_range_m=500.0
    )
    assert not fp.valid
