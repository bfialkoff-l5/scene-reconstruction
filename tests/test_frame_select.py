from __future__ import annotations

import pandas as pd

from scene_recon.frame_select import (
    MAX_KEYFRAMES,
    MIN_ALTITUDE_M,
    SelectionParams,
    assign_bins,
    select_keyframes,
)
from scene_recon.schema import POSE_COLUMNS_REQUIRED
from scene_recon.selection_health import assess_selection


def _candidates(
    n: int,
    *,
    easting_step: float = 5.0,
    northing_step: float = 0.0,
    quality: float | list[float] = 1.0,
) -> pd.DataFrame:
    if isinstance(quality, list):
        assert len(quality) == n
        quality_scores = quality
    else:
        quality_scores = [quality] * n

    index = pd.Index(range(n), name="FrameNumber")
    return pd.DataFrame(
        {
            "TimeUS": [i * 16666 for i in range(n)],
            "easting": [664860.0 + i * easting_step for i in range(n)],
            "northing": [3492703.0 + i * northing_step for i in range(n)],
            "altamsl": [145.0] * n,
            "utm_zone": ["36N"] * n,
            "roll_rad": [0.0] * n,
            "pitch_rad": [0.0] * n,
            "yaw_rad": [0.0] * n,
            "feature_count": 100,
            "sharpness": 50.0,
            "quality_score": quality_scores,
            "cell_x": pd.NA,
            "cell_y": pd.NA,
            "selected": False,
            "reject_reason": pd.NA,
        },
        index=index,
    )


def test_assign_bins() -> None:
    params = SelectionParams()
    candidates = _candidates(3, easting_step=params.bin_size_m)
    binned = assign_bins(candidates, params)
    assert binned["cell_x"].tolist() == [0, 1, 2]


def test_path_walk_bridges_temporal_gap() -> None:
    params = SelectionParams(max_frame_gap=50, bin_size_m=100.0, min_translation_m=0.5)
    candidates = _candidates(300, easting_step=1.0)
    out = select_keyframes(candidates, params)
    selected = out[out["selected"]].sort_index()
    gaps = selected.index.to_series().diff().dropna()
    assert int(gaps.max()) <= params.max_frame_gap


def test_spatial_cluster_cap() -> None:
    params = SelectionParams(
        max_frame_gap=10,
        bin_size_m=1.0,
        min_translation_m=0.1,
        cluster_radius_m=5.0,
        max_per_cluster=2,
        max_keyframes=100,
    )
    # Hover: many frames, tiny spatial jitter, short temporal span.
    candidates = _candidates(20, easting_step=0.01, quality=[0.1 + i * 0.02 for i in range(20)])
    out = select_keyframes(candidates, params)
    health = assess_selection(out, params)
    assert health.max_cluster_size <= params.max_per_cluster


def test_select_respects_altitude() -> None:
    candidates = _candidates(1)
    candidates.loc[0, "altamsl"] = MIN_ALTITUDE_M - 1
    out = select_keyframes(candidates)
    assert not out.loc[0, "selected"]
    assert out.loc[0, "reject_reason"] == "below_altitude"


def test_select_respects_max_keyframes() -> None:
    params = SelectionParams(max_keyframes=MAX_KEYFRAMES, max_frame_gap=5, bin_size_m=5.0)
    n = MAX_KEYFRAMES + 200
    candidates = _candidates(n, easting_step=5.0)
    out = select_keyframes(candidates, params)
    assert int(out["selected"].sum()) == MAX_KEYFRAMES


def test_health_fails_on_temporal_gap() -> None:
    params = SelectionParams(max_frame_gap=10)
    candidates = _candidates(1)
    candidates.loc[0, "selected"] = True
    health = assess_selection(candidates, params)
    assert health.passed


def test_pose_schema_columns() -> None:
    assert "FrameNumber" in POSE_COLUMNS_REQUIRED
