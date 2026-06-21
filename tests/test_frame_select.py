from __future__ import annotations

import numpy as np
import pandas as pd
from conftest import square_footprints

from scene_recon.schema import POSE_COLUMNS_REQUIRED
from scene_recon.selection import (
    MAX_KEYFRAMES,
    MIN_ALTITUDE_M,
    GroundGrid,
    SelectionParams,
    assign_bins,
    compute_view_counts,
    select_keyframes,
)
from scene_recon.selection.selector import airborne_span
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


def _run(candidates: pd.DataFrame, params: SelectionParams):
    grid = GroundGrid.from_poses(candidates, bin_size_m=params.bin_size_m)
    footprints = square_footprints(candidates, grid)
    out = select_keyframes(candidates, footprints, grid, params)
    return out, footprints


def test_assign_bins() -> None:
    params = SelectionParams()
    candidates = _candidates(3, easting_step=params.bin_size_m)
    binned = assign_bins(candidates, params)
    assert binned["cell_x"].tolist() == [0, 1, 2]


def test_stage1_superset_spacing() -> None:
    """Stage 1 keeps a new frame once the camera has moved >= keyframe_spacing_m, so
    consecutive kept frames are at least that far apart (a real triangulation
    baseline) and dense frames get thinned out."""
    spacing = 8.0
    params = SelectionParams(
        bin_size_m=5.0, max_keyframes=10_000, keyframe_spacing_m=spacing
    )
    candidates = _candidates(60, easting_step=2.0)  # 2 m/frame -> keep ~every 4th
    grid = GroundGrid.from_poses(candidates, bin_size_m=params.bin_size_m)
    footprints = square_footprints(candidates, grid, half_cells=2)
    out = select_keyframes(candidates, footprints, grid, params)
    kept = out[out["selected"]].index.to_list()
    assert 1 < len(kept) < 60, f"expected spacing, kept {len(kept)}/60"
    for a, b in zip(kept, kept[1:]):
        d = abs(candidates.loc[b, "easting"] - candidates.loc[a, "easting"])
        assert d >= spacing - 1e-9, f"consecutive kept {a}->{b} only {d:.1f} m apart"
    others = out[~out["selected"]]["reject_reason"].dropna().unique().tolist()
    assert "redundant_spacing" in others


def test_stage2_best_quality_per_bin() -> None:
    """Stage 2 thins the superset into max_keyframes path bins and keeps the
    highest-quality frame in each."""
    quality = [0.5] * 10
    quality[3] = 1.0  # winner of first half
    quality[8] = 1.0  # winner of second half
    candidates = _candidates(10, easting_step=5.0, quality=quality)
    # spacing 0 keeps every frame in the superset, isolating the Stage-2 thinning
    params = SelectionParams(bin_size_m=5.0, max_keyframes=2, keyframe_spacing_m=0.0)
    out, _ = _run(candidates, params)
    selected = set(out[out["selected"]].index.to_list())
    assert selected == {3, 8}, selected
    assert (out.loc[out["selected"], "selection_reason"] == "keyframe").all()
    assert (out.loc[[0, 1, 2, 4], "reject_reason"] == "thinned_by_budget").all()


def test_coverage_cull_caps_redundancy() -> None:
    from scene_recon.geometry.footprint import GroundFootprint
    from scene_recon.selection.selector import _coverage_cull

    def _fp(f: int) -> GroundFootprint:
        return GroundFootprint(
            frame_number=f, cells=frozenset({(0, 0), (0, 1)}), valid=True,
            valid_frac=1.0, area_m2=0.0, centroid_e=0.0, centroid_n=0.0,
            hull_wkt="", reject_detail=None,
        )

    fps = {i: _fp(i) for i in range(10)}
    quality = pd.Series({i: i / 10.0 for i in range(10)})
    survivors, dropped = _coverage_cull(list(range(10)), fps, quality, floor=3)
    assert set(survivors) == {7, 8, 9}  # 3 highest-quality survive the floor
    assert len(dropped) == 7
    # every cell retains exactly the floor's worth of views
    counts = {}
    for i in survivors:
        for c in fps[i].cells:
            counts[c] = counts.get(c, 0) + 1
    assert all(v == 3 for v in counts.values())


def test_select_respects_altitude() -> None:
    params = SelectionParams()
    candidates = _candidates(1)
    candidates.loc[0, "altamsl"] = MIN_ALTITUDE_M - 1
    out, _ = _run(candidates, params)
    assert not out.loc[0, "selected"]
    assert out.loc[0, "reject_reason"] == "below_altitude"


def test_airborne_span_trims_ground_keeps_low_pass() -> None:
    # rest(100) -> climb(50) -> cruise(80) -> low pass(10) -> cruise(110) -> descent(50) -> rest(100)
    prof = np.concatenate(
        [
            np.full(100, 145.0),
            np.linspace(145, 230, 50),
            np.full(80, 230.0),
            np.full(10, 150.0),
            np.full(110, 230.0),
            np.linspace(230, 145, 50),
            np.full(100, 145.0),
        ]
    )
    mask = airborne_span(prof, 2.0)
    assert not mask[:100].any()  # pre-takeoff ground trimmed
    assert not mask[-100:].any()  # post-landing ground trimmed
    assert mask[240:250].all()  # mid-flight low pass kept (it is airborne)
    # constant altitude (never leaves ground) keeps everything for other gates to judge
    assert airborne_span(np.full(50, 145.0), 2.0).all()


def test_ground_frames_rejected_on_ground() -> None:
    n = 200
    candidates = _candidates(n, easting_step=5.0)
    candidates["altamsl"] = list(np.full(50, 145.0)) + list(np.linspace(145, 230, 30)) + list(np.full(120, 230.0))
    params = SelectionParams(bin_size_m=5.0)
    out, _ = _run(candidates, params)
    assert (out.loc[:49, "reject_reason"] == "on_ground").all()
    assert not out.loc[:49, "selected"].any()


def test_select_respects_max_keyframes() -> None:
    cap = 50
    params = SelectionParams(max_keyframes=cap, bin_size_m=5.0)
    n = cap + 200
    candidates = _candidates(n, easting_step=5.0)
    out, _ = _run(candidates, params)
    assert int(out["selected"].sum()) == cap


def test_invalid_footprint_rejected() -> None:
    params = SelectionParams(bin_size_m=5.0)
    candidates = _candidates(5, easting_step=5.0)
    grid = GroundGrid.from_poses(candidates, bin_size_m=params.bin_size_m)
    footprints = square_footprints(candidates, grid)
    bad = footprints[2]
    footprints[2] = type(bad)(
        frame_number=2,
        cells=frozenset(),
        valid=False,
        valid_frac=0.0,
        area_m2=0.0,
        centroid_e=float("nan"),
        centroid_n=float("nan"),
        hull_wkt="POLYGON EMPTY",
        reject_detail="too_few_rays",
    )
    out = select_keyframes(candidates, footprints, grid, params)
    assert out.loc[2, "reject_reason"] == "invalid_footprint"
    assert not out.loc[2, "selected"]


def test_fine_gsd_outliers_rejected() -> None:
    """A frame flying far lower than the flight median (fine GSD) is dropped by the
    GSD-consistency floor, and a large gsd_ratio_max disables the gate."""
    candidates = _candidates(20, easting_step=5.0)
    candidates["agl_m"] = 60.0  # uniform survey altitude...
    candidates.loc[5, "agl_m"] = 5.0  # ...except one near-ground frame (60/5 = 12x > 3)
    grid = GroundGrid.from_poses(candidates, bin_size_m=5.0)
    footprints = square_footprints(candidates, grid, half_cells=2)

    out = select_keyframes(
        candidates, footprints, grid, SelectionParams(bin_size_m=5.0, gsd_ratio_max=3.0)
    )
    assert out.loc[5, "reject_reason"] == "fine_gsd"
    assert not out.loc[5, "selected"]

    out_off = select_keyframes(
        candidates, footprints, grid, SelectionParams(bin_size_m=5.0, gsd_ratio_max=1e9)
    )
    assert int((out_off["reject_reason"] == "fine_gsd").sum()) == 0


def test_health_passes_single_selection() -> None:
    params = SelectionParams()
    candidates = _candidates(1)
    candidates.loc[0, "selected"] = True
    health = assess_selection(candidates, params)
    assert health.passed


def test_two_stage_spreads_across_mission() -> None:
    """End-to-end: even overlap spacing covers a multi-pass lawnmower mission edge
    to edge (the regression that green unit tests previously missed)."""
    rows = []
    frame = 0
    origin_e, origin_n = 664860.0, 3492703.0
    step = 5.0
    for row in range(4):
        n = 80
        for i in range(n):
            e = origin_e + (i if row % 2 == 0 else n - 1 - i) * step
            n_ = origin_n + row * 40.0
            rows.append(
                {
                    "FrameNumber": frame,
                    "easting": e,
                    "northing": n_,
                    "altamsl": 150.0,
                    "quality_score": 0.8,
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

    params = SelectionParams(max_keyframes=320, bin_size_m=5.0)
    grid = GroundGrid.from_poses(df, bin_size_m=params.bin_size_m)
    footprints = square_footprints(df, grid, half_cells=2)
    out = select_keyframes(df, footprints, grid, params)
    sel = out[out["selected"]]
    assert (sel["selection_reason"] == "keyframe").all()
    e_span = float(sel["easting"].max() - sel["easting"].min())
    n_span = float(sel["northing"].max() - sel["northing"].min())
    assert e_span > 200.0
    assert n_span > 80.0


def test_pose_schema_columns() -> None:
    assert "FrameNumber" in POSE_COLUMNS_REQUIRED
