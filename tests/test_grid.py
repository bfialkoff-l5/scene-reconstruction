from dataclasses import dataclass

import numpy as np
import pandas as pd

from scene_recon.selection.grid import GroundGrid


@dataclass
class StubFootprint:
    cells: frozenset
    valid: bool


def test_from_poses_floored_aligned_and_deterministic():
    df = pd.DataFrame(
        {
            "easting": [1003.0, 1011.0, 1027.5, 1004.2],
            "northing": [2002.5, 2030.0, 2009.0, 2001.0],
        }
    )
    grid = GroundGrid.from_poses(df, bin_size_m=10.0)
    # origin floored to UTM-aligned multiple of bin size, below-left of data
    assert grid.origin_e == 1000.0
    assert grid.origin_n == 2000.0
    assert grid.origin_e <= df["easting"].min()
    assert grid.origin_n <= df["northing"].min()
    # independent of row order
    shuffled = df.iloc[[2, 0, 3, 1]].reset_index(drop=True)
    grid2 = GroundGrid.from_poses(shuffled, bin_size_m=10.0)
    assert (grid.origin_e, grid.origin_n) == (grid2.origin_e, grid2.origin_n)
    # margin pushes origin further below-left
    grid_margin = GroundGrid.from_poses(df, bin_size_m=10.0, margin_m=5.0)
    assert grid_margin.origin_e == 990.0
    assert grid_margin.origin_n == 1990.0


def test_cells_for_points_offsets_dedupe_nan_and_z():
    grid = GroundGrid(bin_size_m=10.0, origin_e=1000.0, origin_n=2000.0)
    # one point in cell (0,0), one in (2,3), and a duplicate in (0,0)
    pts = np.array(
        [
            [1005.0, 2005.0],  # (0, 0)
            [1025.0, 2031.0],  # (2, 3)
            [1001.0, 2009.0],  # (0, 0) duplicate
        ]
    )
    assert grid.cells_for_points(pts) == frozenset({(0, 0), (2, 3)})

    # NaN row dropped
    pts_nan = np.array(
        [
            [1005.0, 2005.0],
            [np.nan, 2031.0],
        ]
    )
    assert grid.cells_for_points(pts_nan) == frozenset({(0, 0)})

    # (N,3) ignores z and matches (N,2)
    pts3 = np.array(
        [
            [1005.0, 2005.0, 99.0],
            [1025.0, 2031.0, -50.0],
            [1001.0, 2009.0, 7.0],
        ]
    )
    assert grid.cells_for_points(pts3) == grid.cells_for_points(pts)

    # empty input
    assert grid.cells_for_points(np.empty((0, 2))) == frozenset()


def test_cell_id_matches_cells_for_points():
    grid = GroundGrid(bin_size_m=10.0, origin_e=1000.0, origin_n=2000.0)
    e, n = 1025.0, 2031.0
    assert grid.cells_for_points(np.array([[e, n]])) == frozenset({grid.cell_id(e, n)})


def test_mission_cells_unions_valid_only():
    grid = GroundGrid(bin_size_m=10.0, origin_e=0.0, origin_n=0.0)
    a = StubFootprint(cells=frozenset({(0, 0), (1, 1)}), valid=True)
    b = StubFootprint(cells=frozenset({(1, 1), (2, 2)}), valid=True)
    bad = StubFootprint(cells=frozenset({(9, 9)}), valid=False)
    assert grid.mission_cells([a, b, bad]) == frozenset({(0, 0), (1, 1), (2, 2)})
    assert grid.mission_cells([]) == frozenset()


def test_cell_center_round_trips():
    grid = GroundGrid(bin_size_m=10.0, origin_e=1000.0, origin_n=2000.0)
    for c in [(0, 0), (2, 3), (-1, -5), (10, 7)]:
        e, n = grid.cell_center(c)
        assert grid.cell_id(e, n) == c
