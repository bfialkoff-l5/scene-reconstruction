from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scene_recon.selection import DEFAULT_SELECTION_PARAMS, SelectionParams, select_keyframes
from scene_recon.selection_health import assess_selection
from scene_recon.selection_report import build_audit_df, build_summary, write_selection_report


def test_build_summary_includes_health(tmp_path: Path) -> None:
    params = SelectionParams(min_pct_cells_at_target=0.0)
    index = pd.Index([0, 1, 2], name="FrameNumber")
    candidates = pd.DataFrame(
        {
            "easting": [0.0, 5.0, 10.0],
            "northing": [0.0, 0.0, 0.0],
            "altamsl": [10.0, 10.0, 10.0],
            "roll_rad": [0.0, 0.0, 0.0],
            "pitch_rad": [0.0, 0.0, 0.0],
            "yaw_rad": [0.0, 0.0, 0.0],
            "quality_score": [0.5, 0.9, 0.7],
            "cell_x": [0, 1, 2],
            "cell_y": [0, 0, 0],
            "selected": [True, True, True],
            "reject_reason": [pd.NA, pd.NA, pd.NA],
        },
        index=index,
    )
    constants = params.as_constants()
    health = assess_selection(candidates, params)
    summary = build_summary(candidates, constants, health=health)
    assert summary["health"]["passed"]


def test_write_selection_report(tmp_path: Path) -> None:
    params = SelectionParams(max_frame_gap=100, bin_size_m=5.0)
    candidates = select_keyframes(
        pd.DataFrame(
            {
                "easting": [0.0, 5.0, 10.0, 15.0],
                "northing": [0.0, 0.0, 0.0, 0.0],
                "altamsl": [10.0] * 4,
                "quality_score": [0.4, 0.9, 0.8, 0.6],
                "roll_rad": [0.0] * 4,
                "pitch_rad": [0.0] * 4,
                "yaw_rad": [0.0] * 4,
                "cell_x": pd.NA,
                "cell_y": pd.NA,
                "selected": False,
                "reject_reason": pd.NA,
            },
            index=pd.Index(range(4), name="FrameNumber"),
        ),
        params,
    )
    constants = params.as_constants()
    health = assess_selection(candidates, params)
    write_selection_report(candidates, tmp_path, constants, health=health)
    assert (tmp_path / "selection_audit.csv").is_file()
    assert (tmp_path / "selection_summary.json").is_file()
    assert (tmp_path / "selection_report" / "trajectory_map.png").is_file()
    audit = build_audit_df(candidates)
    assert "bin_rank" in audit.columns
    summary = json.loads((tmp_path / "selection_summary.json").read_text())
    assert "health" in summary
