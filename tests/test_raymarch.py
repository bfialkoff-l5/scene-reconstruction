from __future__ import annotations

import numpy as np
from affine import Affine

from scene_recon.geometry.terrain import TerrainModel
from scene_recon.geometry.raymarch import raymarch_first_hit

# 200x200 grid, 10 m cells. world_x = -500 + 10*col, world_y = 1500 - 10*row.
# Covers easting [-500, 1490], northing [-490, 1500] -> contains the test rays.
TRANSFORM = Affine.translation(-500, 1500) * Affine.scale(10, -10)


def _flat(z=0.0):
    return TerrainModel.from_array(np.full((200, 200), z, dtype=np.float32), TRANSFORM)


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def test_straight_down_hits_origin_xy():
    terrain = _flat(0.0)
    origins = np.array([[0.0, 0.0, 100.0]])
    directions = np.array([[0.0, 0.0, -1.0]])
    hits, valid = raymarch_first_hit(origins, directions, terrain)
    assert valid[0]
    assert np.allclose(hits[0], [0.0, 0.0, 0.0], atol=1e-6)


def test_oblique_ray_hits_far_ground():
    terrain = _flat(0.0)
    origins = np.array([[0.0, 0.0, 100.0]])
    directions = np.array([_unit((1.0, 0.0, -1.0))])
    hits, valid = raymarch_first_hit(origins, directions, terrain)
    assert valid[0]
    assert np.allclose(hits[0], [100.0, 0.0, 0.0], atol=1.0)


def test_wall_occludes_far_ground():
    array = np.zeros((200, 200), dtype=np.float32)
    # wall of height 50 spanning easting ~[40, 90] (cols 54..59), all rows.
    array[:, 54:60] = 50.0
    terrain = TerrainModel.from_array(array, TRANSFORM)

    # gentle descent: above ground before the wall, crosses the wall top first.
    origins = np.array([[0.0, 0.0, 55.0]])
    directions = np.array([_unit((1.0, 0.0, -0.1))])
    hits, valid = raymarch_first_hit(origins, directions, terrain)
    assert valid[0]
    assert abs(hits[0, 2] - 50.0) < 1.5
    assert hits[0, 0] < 90.0  # hit the wall, not the far ground at x~550


def test_up_looking_ray_is_invalid():
    terrain = _flat(0.0)
    origins = np.array([[0.0, 0.0, 100.0]])
    directions = np.array([[0.0, 0.0, 1.0]])
    hits, valid = raymarch_first_hit(origins, directions, terrain)
    assert not valid[0]


def test_batch_mixed_validity():
    terrain = _flat(0.0)
    origins = np.array([[0.0, 0.0, 100.0], [0.0, 0.0, 100.0]])
    directions = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    hits, valid = raymarch_first_hit(origins, directions, terrain)
    assert valid[0] and not valid[1]
    assert np.allclose(hits[0], [0.0, 0.0, 0.0], atol=1e-6)
