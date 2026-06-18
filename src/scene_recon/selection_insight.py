"""Coverage-quality insight beyond raw view counts: where the gaps are, where
budget was wasted, and whether covered cells have reconstructable view geometry."""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from scene_recon.selection import FootprintCache, GroundGrid, ViewCounts
from scene_recon.selection.params import SelectionParams


def coverage_holes(
    view_counts: ViewCounts,
    mission_cells: frozenset[tuple[int, int]],
    grid: GroundGrid,
    target: int,
    *,
    top_n: int = 10,
) -> dict:
    under = {c for c in mission_cells if view_counts.get(c, 0) < target}
    seen: set[tuple[int, int]] = set()
    holes: list[dict] = []
    for start in under:
        if start in seen:
            continue
        comp: list[tuple[int, int]] = []
        queue = deque([start])
        seen.add(start)
        while queue:
            cx, cy = queue.popleft()
            comp.append((cx, cy))
            for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                if (nx, ny) in under and (nx, ny) not in seen:
                    seen.add((nx, ny))
                    queue.append((nx, ny))
        centers = [grid.cell_center(c) for c in comp]
        holes.append(
            {
                "n_cells": len(comp),
                "area_m2": round(len(comp) * grid.bin_size_m**2, 1),
                "centroid_e": round(float(np.mean([e for e, _ in centers])), 1),
                "centroid_n": round(float(np.mean([n for _, n in centers])), 1),
            }
        )
    holes.sort(key=lambda h: h["n_cells"], reverse=True)
    return {
        "n_holes": len(holes),
        "n_under_target_cells": len(under),
        "total_hole_area_m2": round(len(under) * grid.bin_size_m**2, 1),
        "largest_holes": holes[:top_n],
    }


def compute_insight(
    candidates: pd.DataFrame,
    footprints: FootprintCache,
    view_counts: ViewCounts,
    grid: GroundGrid,
    params: SelectionParams,
    *,
    cell_ground_z: dict[tuple[int, int], float] | None = None,
) -> tuple[dict, dict[tuple[int, int], float]]:
    mission_cells = grid.mission_cells(footprints.values())
    holes = coverage_holes(view_counts, mission_cells, grid, params.target_views_per_cell)
    if not candidates["selected"].any():
        return {"coverage_holes": holes}, {}

    from scene_recon.selection.parallax import (
        approx_cell_ground_z,
        build_parallax_context,
        cell_viewers_from_selection,
        convergence_by_cell,
        parallax_metrics,
    )

    ground_z = cell_ground_z or approx_cell_ground_z(grid, mission_cells, candidates)
    ctx = build_parallax_context(candidates)
    viewers = cell_viewers_from_selection(candidates, footprints)
    conv_by_cell = convergence_by_cell(viewers, ctx, grid, ground_z)
    metrics = parallax_metrics(viewers, ctx, grid, ground_z, params)
    threshold = params.parallax_min_convergence_deg
    multi_view = [
        conv_by_cell[c]
        for c in conv_by_cell
        if len(viewers[c]) >= params.parallax_min_views
    ]
    parallax = {
        **metrics.as_dict(),
        "parallax_min_views": params.parallax_min_views,
        "parallax_min_convergence_deg": threshold,
        "n_cells_multi_view_low_convergence": int(sum(1 for s in multi_view if s < threshold)),
    }
    return {"coverage_holes": holes, "parallax": parallax}, conv_by_cell
