from __future__ import annotations

import numpy as np

from scene_recon.selection import GroundGrid, SelectionParams
from scene_recon.selection.parallax import (
    build_parallax_context,
    convergence_by_cell,
    max_convergence_deg,
    parallax_metrics,
    parallax_satisfied,
)
import pandas as pd


def _out(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, index=pd.Index([r.pop("FrameNumber") for r in rows], name="FrameNumber"))


def test_max_convergence_wide_vs_near_duplicate() -> None:
    wide = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert abs(max_convergence_deg(wide) - 90.0) < 1e-6
    near = np.array([[0.0, 0.0, 1.0], [0.01, 0.0, 0.9999]])
    assert max_convergence_deg(near) < 1.0
    assert max_convergence_deg(np.array([[0.0, 0.0, 1.0]])) == 0.0


def test_convergence_by_cell_and_satisfied() -> None:
    grid = GroundGrid(bin_size_m=10.0, origin_e=0.0, origin_n=0.0)
    params = SelectionParams(parallax_min_views=3, parallax_min_convergence_deg=10.0)
    # cell (0,0) center (5,5); cameras south and west of it at equal altitude.
    out = _out(
        [
            {"FrameNumber": 0, "easting": 5.0, "northing": -50.0, "altamsl": 100.0},
            {"FrameNumber": 1, "easting": -50.0, "northing": 5.0, "altamsl": 100.0},
        ]
    )
    cell = (0, 0)
    z = {cell: 0.0}
    ctx = build_parallax_context(out)
    viewers = {cell: [0, 1]}
    conv = convergence_by_cell(viewers, ctx, grid, z)
    assert 35.0 < conv[cell] < 50.0
    # 2 views < parallax_min_views=3 => not satisfied even though angle is wide.
    assert not parallax_satisfied([0, 1], cell, ctx, grid, z, params)

    metrics = parallax_metrics(viewers, ctx, grid, z, params)
    assert metrics.n_cells_covered == 1
    assert metrics.n_parallax_satisfied == 0
    assert abs(metrics.median_max_convergence_deg - conv[cell]) < 0.1
