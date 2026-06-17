from __future__ import annotations

import pandas as pd

from scene_recon.record import Record
from scene_recon.schema import POSE_COLUMNS_REQUIRED


def load_poses(record: Record) -> pd.DataFrame:
    df = pd.read_csv(record.poses_path)
    missing = [c for c in POSE_COLUMNS_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"poses CSV missing columns: {missing}")

    df = df.sort_values("FrameNumber").drop_duplicates(subset=["FrameNumber"], keep="first")
    df = df.set_index("FrameNumber")
    df.index.name = "FrameNumber"
    return df
