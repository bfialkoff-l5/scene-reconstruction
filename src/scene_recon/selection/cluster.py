from __future__ import annotations

import numpy as np
import pandas as pd

from scene_recon.selection.params import SelectionParams


def max_local_density(
    indices: list[int],
    out: pd.DataFrame,
    params: SelectionParams,
) -> tuple[int, list[int]]:
    """Densest spatial cluster among selected frames (health reporting only).

    Returns (size, member frame numbers) of the largest set of selected frames
    within ``cluster_radius_m`` of any one member. ponytail: O(n^2) brute force;
    fine for keyframe counts (<= a few thousand). Upgrade to a KD-tree if it ever
    dominates report time.
    """
    if not indices:
        return 0, []
    e = out.loc[indices, "easting"].astype(float).to_numpy()
    n = out.loc[indices, "northing"].astype(float).to_numpy()
    radius_sq = params.cluster_radius_m * params.cluster_radius_m
    best_count = 0
    best_members: list[int] = []
    for i in range(len(indices)):
        de = e - e[i]
        dn = n - n[i]
        in_ball = np.flatnonzero(de * de + dn * dn <= radius_sq)
        if in_ball.size > best_count:
            best_count = int(in_ball.size)
            best_members = [int(indices[j]) for j in in_ball]
    return best_count, best_members
