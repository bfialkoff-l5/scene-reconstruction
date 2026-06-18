from __future__ import annotations

import pandas as pd
from conftest import square_footprints

from scene_recon.selection import (
    GroundGrid,
    SelectionParams,
    compute_view_counts,
    select_keyframes,
)
from scene_recon.selection_health import SelectionFailed, assess_selection


def _hover_tail_candidates() -> pd.DataFrame:
    rows = []
    frame = 0
    transit_end_e = 664860.0 + 199 * 5.0
    for i in range(200):
        rows.append(
            {
                "FrameNumber": frame,
                "easting": 664860.0 + i * 5.0,
                "northing": 3492703.0,
                "altamsl": 150.0,
                "quality_score": 0.8,
            }
        )
        frame += 1
    for i in range(80):
        rows.append(
            {
                "FrameNumber": frame,
                "easting": transit_end_e + (i % 5) * 0.5,
                "northing": 3492703.0 + (i % 3) * 0.4,
                "altamsl": 145.0,
                "quality_score": 0.95,
            }
        )
        frame += 1

    df = pd.DataFrame(rows).set_index("FrameNumber")
    for col, val in [
        ("TimeUS", 0),
        ("utm_zone", "36N"),
        ("roll_rad", 0.0),
        ("pitch_rad", 0.0),
        ("yaw_rad", 0.0),
        ("feature_count", 100),
        ("sharpness", 50.0),
        ("cell_x", pd.NA),
        ("cell_y", pd.NA),
        ("selected", False),
        ("reject_reason", pd.NA),
    ]:
        df[col] = val
    return df


def test_hover_tail_capped_and_covered() -> None:
    params = SelectionParams(bin_size_m=3.0, cluster_radius_m=10.0, max_keyframes=500)
    candidates = _hover_tail_candidates()
    grid = GroundGrid.from_poses(candidates, bin_size_m=params.bin_size_m)
    footprints = square_footprints(candidates, grid)
    out = select_keyframes(candidates, footprints, grid, params)
    view_counts = compute_view_counts(out, footprints)
    mission_cells = grid.mission_cells(footprints.values())
    health = assess_selection(
        out, params, view_counts=view_counts, mission_cells=mission_cells
    )
    selected = out[out["selected"]]
    assert 0 < len(selected) <= params.max_keyframes
    assert (selected["selection_reason"] == "keyframe").all()
    assert health.passed


def test_selection_health_includes_parallax_when_footprints_given() -> None:
    params = SelectionParams(max_keyframes=50)
    candidates = _hover_tail_candidates()
    grid = GroundGrid.from_poses(candidates, bin_size_m=params.bin_size_m)
    footprints = square_footprints(candidates, grid)
    out = select_keyframes(candidates, footprints, grid, params)
    view_counts = compute_view_counts(out, footprints)
    mission_cells = grid.mission_cells(footprints.values())
    health = assess_selection(
        out,
        params,
        view_counts=view_counts,
        mission_cells=mission_cells,
        footprints=footprints,
        grid=grid,
    )
    assert health.parallax is not None
    assert health.parallax.n_cells_covered > 0
    assert health.passed


def test_selection_failed_message() -> None:
    health = assess_selection(
        pd.DataFrame(
            {
                "selected": [False],
                "easting": [0.0],
                "northing": [0.0],
            },
            index=pd.Index([0], name="FrameNumber"),
        ),
        SelectionParams(),
    )
    assert not health.passed
    exc = SelectionFailed(health)
    assert "no frames selected" in str(exc)
