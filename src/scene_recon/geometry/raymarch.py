from __future__ import annotations

import numpy as np


def raymarch_first_hit(
    origins,
    directions,
    terrain,
    *,
    step_m: float = 10.0,
    max_range_m: float = 2000.0,
    refine_iters: int = 4,
):
    """March each ray to its first terrain crossing (occlusion-correct).

    Returns (hits (N,3) float, valid (N,) bool). Vectorized across rays;
    the only python loop is over march steps.
    """
    origins = np.asarray(origins, dtype=float)
    directions = np.asarray(directions, dtype=float)
    n = origins.shape[0]

    hits = np.full((n, 3), np.nan)
    valid = np.zeros(n, dtype=bool)
    done = np.zeros(n, dtype=bool)

    def f_at(t):
        """Signed height (point_z - terrain_z) and terrain_z at param t (scalar or (N,))."""
        p = origins + np.asarray(t).reshape(-1, 1) * directions
        tz = terrain.elevation_at(p[:, 0], p[:, 1])
        return p[:, 2] - tz, tz

    f_low, tz_low = f_at(0.0)
    done |= f_low <= 0  # started already below terrain (nan -> False)

    t = 0.0
    while t < max_range_m:
        t_high = min(t + step_m, max_range_m)
        f_high, tz_high = f_at(t_high)

        cross = (
            (~done)
            & (f_low > 0)
            & (f_high <= 0)
            & np.isfinite(tz_low)
            & np.isfinite(tz_high)
        )
        if cross.any():
            a = np.full(n, t)
            b = np.full(n, t_high)
            fa = f_low.copy()
            for _ in range(refine_iters):
                m = 0.5 * (a + b)
                fm, _ = f_at(m)
                right = fm > 0
                a = np.where(right, m, a)
                fa = np.where(right, fm, fa)
                b = np.where(right, b, m)
            fb, _ = f_at(b)
            denom = fa - fb
            t_hit = np.where(np.abs(denom) > 1e-12, a + fa * (b - a) / denom, 0.5 * (a + b))
            p_hit = origins + t_hit.reshape(-1, 1) * directions

            hits[cross] = p_hit[cross]
            valid[cross] = True
            done[cross] = True

        f_low, tz_low = f_high, tz_high
        t = t_high

    return hits, valid
