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
    z_bounds: tuple[float, float] | None = None,
):
    """March each ray to its first terrain crossing (occlusion-correct).

    Returns (hits (N,3) float, valid (N,) bool). Vectorized across rays; the only
    python loop is over march steps.

    Smart marching (when ``z_bounds=(z_lo, z_hi)`` is the DTM elevation band):
    terrain only exists inside that band, so each ray can only hit between where it
    crosses the ``z_hi`` plane (enters the band) and the ``z_lo`` plane (leaves it).
    We march that per-ray bracket instead of 0..max_range, cull non-descending
    "sky" rays, retire rays once they drop below the band, and stop sampling rays
    that already hit. The step grid stays anchored to multiples of ``step_m``, so
    results are identical to the naive full-range march.
    """
    origins = np.asarray(origins, dtype=float)
    directions = np.asarray(directions, dtype=float)
    n = origins.shape[0]

    hits = np.full((n, 3), np.nan)
    valid = np.zeros(n, dtype=bool)
    if n == 0:
        return hits, valid
    done = np.zeros(n, dtype=bool)

    dz = directions[:, 2]
    oz = origins[:, 2]
    t_exit = np.full(n, max_range_m)

    have_band = (
        z_bounds is not None and np.isfinite(z_bounds[0]) and np.isfinite(z_bounds[1])
    )
    if have_band:
        z_lo, z_hi = float(z_bounds[0]), float(z_bounds[1])
        descending = dz < -1e-9
        with np.errstate(divide="ignore", invalid="ignore"):
            t_a = (z_hi - oz) / dz  # crosses top-of-band plane
            t_b = (z_lo - oz) / dz  # crosses bottom-of-band plane
        t_enter = np.clip(np.minimum(t_a, t_b), 0.0, max_range_m)
        t_exit = np.clip(np.maximum(t_a, t_b), 0.0, max_range_m)
        done |= ~descending  # rays not pointing down can't hit ground below
        active = ~done
        if not active.any():
            return hits, valid
        t = float(np.floor(t_enter[active].min() / step_m) * step_m)
        t_end = float(min(max_range_m, t_exit[active].max()))
    else:
        done |= dz >= 0  # sky rays never hit
        t = 0.0
        t_end = max_range_m

    # signed height (point_z - terrain_z) at the low end of the current step,
    # carried per ray across steps so each step only samples the far edge.
    def height_at(t_at, idx):
        p = origins[idx] + np.asarray(t_at).reshape(-1, 1) * directions[idx]
        tz = terrain.elevation_at(p[:, 0], p[:, 1])
        return p[:, 2] - tz, tz

    f_low = np.empty(n)
    tz_low = np.empty(n)
    f_low[:], tz_low[:] = height_at(t, slice(None))
    done |= f_low <= 0  # already at/below terrain at the window start

    while t < t_end and not done.all():
        t_high = min(t + step_m, max_range_m)
        done |= (~done) & (t > t_exit + 1e-9)  # fully past the band -> can't hit
        idx = np.flatnonzero(~done)
        if idx.size == 0:
            break
        f_h, tz_h = height_at(t_high, idx)
        fl = f_low[idx]
        cross = (fl > 0) & (f_h <= 0) & np.isfinite(tz_low[idx]) & np.isfinite(tz_h)
        if cross.any():
            ci = idx[cross]
            a = np.full(ci.size, t)
            b = np.full(ci.size, t_high)
            fa = f_low[ci].copy()
            for _ in range(refine_iters):
                m = 0.5 * (a + b)
                fm, _ = height_at(m, ci)
                right = fm > 0
                a = np.where(right, m, a)
                fa = np.where(right, fm, fa)
                b = np.where(right, b, m)
            fb, _ = height_at(b, ci)
            denom = fa - fb
            t_hit = np.where(np.abs(denom) > 1e-12, a + fa * (b - a) / denom, 0.5 * (a + b))
            hits[ci] = origins[ci] + t_hit.reshape(-1, 1) * directions[ci]
            valid[ci] = True
            done[ci] = True

        f_low[idx] = f_h
        tz_low[idx] = tz_h
        t = t_high

    return hits, valid
