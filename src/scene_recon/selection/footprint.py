from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from scene_recon.selection.params import SelectionParams

if TYPE_CHECKING:
    from scene_recon.geometry.footprint import GroundFootprint

ViewCounts = dict[tuple[int, int], int]
FootprintCache = dict[int, "GroundFootprint"]


def footprint_jaccard(
    a: frozenset[tuple[int, int]], b: frozenset[tuple[int, int]]
) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def _wrap_angle(rad: float) -> float:
    return (rad + math.pi) % (2 * math.pi) - math.pi


def translation_m(a: pd.Series, b: pd.Series) -> float:
    de = float(a["easting"] - b["easting"])
    dn = float(a["northing"] - b["northing"])
    return math.hypot(de, dn)


def rotation_deg(a: pd.Series, b: pd.Series) -> float:
    dr = _wrap_angle(float(a.get("roll_rad", 0.0) - b.get("roll_rad", 0.0)))
    dp = _wrap_angle(float(a.get("pitch_rad", 0.0) - b.get("pitch_rad", 0.0)))
    dy = _wrap_angle(float(a.get("yaw_rad", 0.0) - b.get("yaw_rad", 0.0)))
    return math.degrees(math.sqrt(dr * dr + dp * dp + dy * dy))


def assign_bins(candidates: pd.DataFrame, params: SelectionParams) -> pd.DataFrame:
    """Per-frame audit cell coordinates (used only for `bin_rank` reporting)."""
    out = candidates.copy()
    origin_e = out["easting"].min()
    origin_n = out["northing"].min()
    out["cell_x"] = np.floor((out["easting"] - origin_e) / params.bin_size_m).astype("Int64")
    out["cell_y"] = np.floor((out["northing"] - origin_n) / params.bin_size_m).astype("Int64")
    return out


def compute_view_counts(candidates: pd.DataFrame, footprints: FootprintCache) -> ViewCounts:
    selected = candidates[candidates["selected"]]
    return _count_cells((int(i) for i in selected.index), footprints)


def _count_cells(frames, footprints: FootprintCache) -> ViewCounts:
    counts: ViewCounts = {}
    for idx in frames:
        fp = footprints.get(int(idx))
        if fp is None:
            continue
        for c in fp.cells:
            counts[c] = counts.get(c, 0) + 1
    return counts


def save_footprints(path: Path, footprints: FootprintCache) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(footprints, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_footprints(path: Path) -> FootprintCache:
    with path.open("rb") as handle:
        return pickle.load(handle)
