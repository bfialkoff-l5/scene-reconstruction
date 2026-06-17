from __future__ import annotations

import pandas as pd

from scene_recon.frame_select import SelectionParams, select_keyframes
from scene_recon.selection_health import SelectionFailed, assess_selection


def _hover_tail_candidates() -> pd.DataFrame:
    rows = []
    frame = 0
    # Transit along a line.
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
    # Hover with GPS jitter.
    for i in range(80):
        rows.append(
            {
                "FrameNumber": frame,
                "easting": 664900.0 + (i % 5) * 0.5,
                "northing": 3492750.0 + (i % 3) * 0.4,
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


def test_hover_tail_capped_and_connected() -> None:
    params = SelectionParams(
        max_frame_gap=30,
        bin_size_m=3.0,
        cluster_radius_m=10.0,
        max_per_cluster=3,
        max_keyframes=500,
    )
    out = select_keyframes(_hover_tail_candidates(), params)
    health = assess_selection(out, params)
    assert health.passed
    assert health.max_cluster_size <= params.max_per_cluster
    assert health.max_temporal_gap <= params.max_frame_gap


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
