from __future__ import annotations

import numpy as np

from scene_recon.odm import recommend_matcher_neighbors


def test_recommend_matcher_neighbors_holds_reach_as_density_changes() -> None:
    """k must grow when spacing densifies, so the matching baseline reach stays fixed.
    On a straight track at spacing s an interior frame sees floor(reach/s) frames each
    side, so the median count is ~2*floor(reach/s)."""
    R = 33.0
    n = np.zeros(100)
    k4 = recommend_matcher_neighbors(np.arange(100) * 4.0, n, reach_m=R)
    k2 = recommend_matcher_neighbors(np.arange(100) * 2.0, n, reach_m=R)
    assert k4 == 2 * int(R // 4.0)  # 16
    assert k2 == 2 * int(R // 2.0)  # 32
    assert k2 > k4  # denser spacing -> more neighbours to span the same reach
    assert recommend_matcher_neighbors([0.0], [0.0]) == 0  # degenerate: no pairs
