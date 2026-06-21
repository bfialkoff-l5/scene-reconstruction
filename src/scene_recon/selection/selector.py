from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from scene_recon.selection.footprint import (
    FootprintCache,
    assign_bins,
)
from scene_recon.selection.grid import GroundGrid
from scene_recon.selection.params import DEFAULT_SELECTION_PARAMS, SelectionParams

log = logging.getLogger(__name__)


def _init_debug_columns(out: pd.DataFrame) -> pd.DataFrame:
    out["selected"] = False
    out["reject_reason"] = pd.NA
    out["selection_reason"] = pd.NA
    out["footprint_area_m2"] = pd.NA
    out["footprint_cell_count"] = pd.NA
    out["footprint_valid"] = pd.NA
    out["footprint_valid_frac"] = pd.NA
    out["footprint_polygon_wkt"] = pd.NA
    return out


def airborne_span(altamsl: np.ndarray, rise_m: float) -> np.ndarray:
    """Boolean per-frame mask (in temporal/index order) that is True between the
    takeoff and landing knees, trimming the stationary pre-takeoff / post-landing
    ground segments while keeping every airborne frame in between (including low
    climb/descent frames and mid-flight low passes).

    Knee = first/last frame whose altitude has risen ``rise_m`` (or 5x the resting
    noise, whichever is larger) above the resting level, sustained over a short
    window so a single noisy spike cannot trip it. Measuring the rise relative to
    the resting altitude cancels the absolute datum.

    ponytail: uses raw altitude, not AGL, so it assumes terrain under the
    resting/takeoff/landing segment is ~flat (verified on 0088: identical knee).
    Upgrade path if a flight takes off across steep terrain: subtract DTM
    elevation per frame before calling.
    """
    a = np.asarray(altamsl, dtype=float)
    if a.size == 0:
        return np.ones(0, dtype=bool)
    rest = np.nanmin(a)
    low = a[a < np.nanpercentile(a, 5)]
    noise = float(np.nanstd(low)) if low.size else 0.0
    delta = max(rise_m, 5 * noise)
    airborne = a > rest + delta
    if not airborne.any():
        return np.ones(a.size, dtype=bool)  # never leaves the ground: keep all, let other gates decide

    # Keep everything between the first and last *sustained* airborne run; runs
    # shorter than min_run are altitude spikes/jitter, not flight, and ignored.
    min_run = min(60, a.size)
    positions = np.flatnonzero(airborne)
    runs = np.split(positions, np.flatnonzero(np.diff(positions) > 1) + 1)
    long_runs = [r for r in runs if r.size >= min_run] or runs
    first = int(long_runs[0][0])
    last = int(long_runs[-1][-1])
    mask = np.zeros(a.size, dtype=bool)
    mask[first : last + 1] = True
    return mask


def _write_footprint_columns(out: pd.DataFrame, footprints: FootprintCache) -> None:
    for idx, fp in footprints.items():
        if idx not in out.index:
            continue
        out.loc[idx, "footprint_valid"] = bool(fp.valid)
        out.loc[idx, "footprint_valid_frac"] = fp.valid_frac
        out.loc[idx, "footprint_polygon_wkt"] = fp.hull_wkt
        out.loc[idx, "footprint_area_m2"] = fp.area_m2
        out.loc[idx, "footprint_cell_count"] = len(fp.cells)


def _distance_superset(
    frames: list[int], pos: dict[int, tuple[float, float]], spacing_m: float
) -> list[int]:
    """Stage 1: walk eligible frames in time order, keep the first, then keep the
    next frame once the camera has moved ``>= spacing_m`` from the last kept frame.
    The distance between consecutive keepers is the triangulation baseline, so this
    guarantees parallax (b/d = spacing/depth) and naturally adapts: a hovering/slow
    segment is thinned to almost nothing, a fast traverse keeps every qualifying
    frame, and a block-boundary teleport (> spacing) always starts a fresh keeper.

    Uses camera position, not footprint overlap, on purpose: footprint Jaccard at
    oblique pitch is dominated by attitude noise (see params.KEYFRAME_SPACING_M) and
    selected zero-baseline near-duplicates that SfM cannot triangulate."""
    superset = [frames[0]]
    lx, ly = pos[frames[0]]
    for idx in frames[1:]:
        x, y = pos[idx]
        if (x - lx) ** 2 + (y - ly) ** 2 >= spacing_m * spacing_m:
            superset.append(idx)
            lx, ly = x, y
    return superset


def _coverage_cull(
    frames: list[int], footprints: FootprintCache, quality: pd.Series, floor: int
) -> tuple[list[int], list[int]]:
    """Drop redundant frames while every footprint cell keeps > ``floor`` views from
    the survivors. Worst-quality frames are considered first so the sharpest survive.
    Returns (survivors_in_path_order, dropped)."""
    counts: dict[tuple[int, int], int] = {}
    for i in frames:
        for c in footprints[i].cells:
            counts[c] = counts.get(c, 0) + 1
    kept = set(frames)
    dropped: list[int] = []
    for i in sorted(frames, key=lambda j: float(quality.loc[j])):
        cells = footprints[i].cells
        if cells and all(counts[c] > floor for c in cells):
            kept.discard(i)
            dropped.append(i)
            for c in cells:
                counts[c] -= 1
    return [i for i in frames if i in kept], dropped


def _thin_to_cap(superset: list[int], quality: pd.Series, cap: int) -> list[int]:
    """Stage 2: partition the path-ordered superset into ``cap`` contiguous bins and
    keep the highest-quality frame in each. Uniform coverage is preserved by
    construction; image quality is maximized per neighborhood. Parallax does not
    degrade because the thinning stays spatially uniform."""
    if len(superset) <= cap:
        return superset
    bins = np.array_split(np.array(superset), cap)
    return [int(max(b, key=lambda i: float(quality.loc[i]))) for b in bins]


def select_keyframes(
    candidates: pd.DataFrame,
    footprints: FootprintCache,
    grid: GroundGrid,
    params: SelectionParams | None = None,
) -> pd.DataFrame:
    p = params or DEFAULT_SELECTION_PARAMS
    out = assign_bins(candidates, p)
    out = _init_debug_columns(out)
    _write_footprint_columns(out, footprints)

    # --- Stage 0: eligibility gate (ineligible frames cannot be selected) ---
    above_floor = out["altamsl"] >= p.min_altitude_m
    out.loc[~above_floor, "reject_reason"] = "below_altitude"

    ordered = out.sort_index()
    airborne = pd.Series(True, index=out.index)
    airborne.loc[ordered.index] = airborne_span(
        ordered["altamsl"].to_numpy(dtype=float), p.ground_trim_rise_m
    )
    out.loc[above_floor & ~airborne, "reject_reason"] = "on_ground"

    eligible = above_floor & airborne

    scored = (
        out["quality_score"].notna()
        if "quality_score" in out.columns
        else pd.Series(False, index=out.index)
    )
    out.loc[eligible & ~scored, "reject_reason"] = "missing_quality_score"

    valid_fp = pd.Series(
        [bool(footprints[i].valid) if i in footprints else False for i in out.index],
        index=out.index,
    )
    out.loc[eligible & scored & ~valid_fp, "reject_reason"] = "invalid_footprint"
    usable = eligible & scored & valid_fp

    # GSD-consistency floor (Goesele: keep matched views within a small resolution
    # ratio, ~1<=r<2). GSD is proportional to AGL, so drop frames flying lower than
    # the flight median by more than gsd_ratio_max -- the near-ground frames whose
    # sub-cm GSD poisons ODM's DSM sizing (a 1 cm DSM over a 410 m site -> OOM).
    # Self-calibrating to the flight's own median; a large ratio disables the gate.
    # Needs the per-frame agl_m column (added by build.py from the DTM); absent it
    # (unit tests, report re-render) the gate is a no-op.
    if usable.any() and "agl_m" in out.columns:
        agl = pd.to_numeric(out["agl_m"], errors="coerce")
        floor = float(agl[usable].median()) / p.gsd_ratio_max
        fine = usable & (agl < floor)
        out.loc[fine, "reject_reason"] = "fine_gsd"
        usable = usable & ~fine
        if fine.any():
            log.info(
                "GSD floor: dropped %d frames below AGL %.1f m (median AGL/%.1f)",
                int(fine.sum()),
                floor,
                p.gsd_ratio_max,
            )

    pool = out.loc[usable].sort_index()
    if pool.empty:
        log.warning("no eligible frames after gate; nothing selected")
        return out

    frames = [int(i) for i in pool.index]

    quality = out["quality_score"].astype(float)

    # --- Stage 1: no-cap distance-spacing superset (baseline = keyframe spacing) ---
    pos = {
        int(i): (float(e), float(n))
        for i, e, n in zip(pool.index, pool["easting"], pool["northing"])
    }
    superset = _distance_superset(frames, pos, p.keyframe_spacing_m)

    # --- Stage 1b (optional): cull over-covered redundant frames ---
    over_covered: list[int] = []
    culled = superset
    if p.max_views_per_cell > 0:
        culled, over_covered = _coverage_cull(
            superset, footprints, quality, p.max_views_per_cell
        )

    # --- Stage 2: thin to the hard budget cap (best quality per path bin) ---
    keep = _thin_to_cap(culled, quality, p.max_keyframes)

    keep_set = set(keep)
    culled_set = set(culled)
    superset_set = set(superset)
    out.loc[keep, "selected"] = True
    out.loc[keep, "selection_reason"] = "keyframe"
    out.loc[keep, "reject_reason"] = pd.NA

    thinned = [i for i in culled if i not in keep_set]
    out.loc[thinned, "reject_reason"] = "thinned_by_budget"
    out.loc[over_covered, "reject_reason"] = "over_covered"
    redundant = [i for i in frames if i not in superset_set]
    out.loc[redundant, "reject_reason"] = "redundant_spacing"

    log.info(
        "selection: %d eligible -> %d superset (spacing %.1f m) -> %d after cull "
        "(>%d views/cell) -> %d keyframes (cap %d)",
        len(frames),
        len(superset),
        p.keyframe_spacing_m,
        len(culled_set),
        p.max_views_per_cell,
        len(keep),
        p.max_keyframes,
    )
    return out
