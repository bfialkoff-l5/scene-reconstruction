from __future__ import annotations

import math

import numpy as np
import pytest
from affine import Affine

from scene_recon.geometry.terrain import TerrainModel

# pixel (col,row) -> world (10*col, 100-10*row): 10 m cells, north-down rows.
TRANSFORM = Affine.translation(0, 100) * Affine.scale(10, -10)


def _model(values, nodata=-9999.0):
    return TerrainModel.from_array(np.asarray(values, dtype=np.float32), TRANSFORM, nodata=nodata)


def test_cell_center_returns_cell_value():
    m = _model([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # pixel (1,1) -> world (10, 90), value array[1,1] == 5
    assert float(m.elevation_at(10.0, 90.0)) == pytest.approx(5.0, abs=1e-6)


def test_bilinear_midpoint_is_average():
    m = _model([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # midpoint between pixels (0,0)=1 and (1,0)=2 -> world (5, 100) -> 1.5
    assert float(m.elevation_at(5.0, 100.0)) == pytest.approx(1.5, abs=1e-6)


def test_out_of_bounds_is_nan():
    m = _model([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert math.isnan(float(m.elevation_at(-100.0, 100.0)))


def test_nodata_cell_is_nan():
    m = _model([[1, 2, 3], [4, -9999.0, 6], [7, 8, 9]])
    # pixel (1,1) is nodata -> world (10, 90)
    assert math.isnan(float(m.elevation_at(10.0, 90.0)))


def test_datum_offset_applied():
    m = TerrainModel.from_array(
        np.array([[1, 2], [3, 4]], dtype=np.float32), TRANSFORM, datum_offset_m=5.0
    )
    assert float(m.elevation_at(0.0, 100.0)) == pytest.approx(6.0, abs=1e-6)


def test_vectorized_shape_preserved():
    m = _model([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    e = np.array([[10.0, 0.0], [20.0, 10.0]])
    n = np.array([[90.0, 100.0], [80.0, 90.0]])
    out = m.elevation_at(e, n)
    assert out.shape == (2, 2)
    assert out[0, 0] == pytest.approx(5.0, abs=1e-6)
