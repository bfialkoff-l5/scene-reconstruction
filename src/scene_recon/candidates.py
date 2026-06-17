from __future__ import annotations

import pandas as pd

from scene_recon.schema import SCORE_COLUMNS, SELECTION_COLUMNS


def init_candidates(poses: pd.DataFrame) -> pd.DataFrame:
    candidates = poses.copy()
    for col in SCORE_COLUMNS:
        candidates[col] = pd.NA
    for col in SELECTION_COLUMNS:
        if col == "selected":
            candidates[col] = False
        else:
            candidates[col] = pd.NA
    return candidates


def selected_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    return candidates[candidates["selected"]].copy()
