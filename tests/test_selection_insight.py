from __future__ import annotations

import pandas as pd

from scene_recon.geometry.footprint import GroundFootprint
from scene_recon.selection import GroundGrid, SelectionParams
from scene_recon.selection.parallax import max_convergence_deg
from scene_recon.selection_insight import compute_insight, coverage_holes


def _fp(frame, cells) -> GroundFootprint:
    return GroundFootprint(
        frame_number=frame,
        cells=frozenset(cells),
        valid=True,
        valid_frac=1.0,
        area_m2=len(cells) * 100.0,
        centroid_e=0.0,
        centroid_n=0.0,
        hull_wkt="",
        reject_detail=None,
    )


def test_coverage_holes_components() -> None:
    grid = GroundGrid(bin_size_m=10.0, origin_e=0.0, origin_n=0.0)
    footprints = {
        0: _fp(0, {(0, 0), (1, 0)}),
        1: _fp(1, {(0, 0), (0, 1)}),
        2: _fp(2, {(5, 5), (5, 6)}),  # unselected region -> a contiguous hole
    }
    mission = grid.mission_cells(footprints.values())
    view_counts = {(0, 0): 2, (1, 0): 1, (0, 1): 1}
    holes = coverage_holes(view_counts, mission, grid, target=2)
    assert holes["n_under_target_cells"] == 4
    assert holes["n_holes"] == 3
    assert holes["largest_holes"][0]["n_cells"] == 2
    assert holes["total_hole_area_m2"] == 400.0


def test_convergence_insight_uses_3d_angle() -> None:
    grid = GroundGrid(bin_size_m=10.0, origin_e=0.0, origin_n=0.0)
    # cell (0,0) center (5,5); cameras south and west at same altitude.
    candidates = pd.DataFrame(
        {
            "easting": [5.0, -50.0],
            "northing": [-50.0, 5.0],
            "altamsl": [100.0, 100.0],
            "selected": [True, True],
        },
        index=pd.Index([0, 1], name="FrameNumber"),
    )
    footprints = {0: _fp(0, {(0, 0)}), 1: _fp(1, {(0, 0)})}
    view_counts = {(0, 0): 2}
    params = SelectionParams(parallax_min_views=3, parallax_min_convergence_deg=10.0)
    insight, conv = compute_insight(
        candidates,
        footprints,
        view_counts,
        grid,
        params,
        cell_ground_z={(0, 0): 0.0},
    )
    assert conv[(0, 0)] > 35.0
    assert conv[(0, 0)] < 50.0
    assert abs(insight["parallax"]["median_max_convergence_deg"] - conv[(0, 0)]) < 0.1
    assert insight["parallax"]["n_parallax_satisfied"] == 0  # only 2 views
    assert "view_diversity" not in insight

    # cross-check against parallax module directly
    import numpy as np

    from scene_recon.selection.parallax import ParallaxContext, cell_direction

    ctx = ParallaxContext(
        {
            0: np.array([5.0, -50.0, 100.0]),
            1: np.array([-50.0, 5.0, 100.0]),
        }
    )
    dirs = np.stack(
        [
            cell_direction(0, (0, 0), ctx, grid, {(0, 0): 0.0}),
            cell_direction(1, (0, 0), ctx, grid, {(0, 0): 0.0}),
        ]
    )
    assert abs(conv[(0, 0)] - max_convergence_deg(dirs)) < 1e-6
