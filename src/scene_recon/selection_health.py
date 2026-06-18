from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from scene_recon.selection import (
    SelectionParams,
    coverage_metrics,
    iter_selection_gaps,
    max_local_density,
)
from scene_recon.selection.footprint import FootprintCache, ViewCounts
from scene_recon.selection.grid import GroundGrid
from scene_recon.selection.metrics import CoverageMetrics
from scene_recon.selection.parallax import (
    CellGroundZ,
    ParallaxMetrics,
    approx_cell_ground_z,
    build_parallax_context,
    parallax_metrics,
)

log = logging.getLogger(__name__)


@dataclass
class SelectionHealth:
    passed: bool
    failures: list[str]
    max_temporal_gap: int
    worst_temporal_gap: dict | None
    max_motion_gap_m: float
    worst_motion_gap: dict | None
    max_cluster_size: int
    largest_cluster: dict | None
    n_selected: int
    coverage: CoverageMetrics
    parallax: ParallaxMetrics | None = None

    def as_dict(self) -> dict:
        out = {
            "passed": self.passed,
            "failures": self.failures,
            "max_temporal_gap": self.max_temporal_gap,
            "worst_temporal_gap": self.worst_temporal_gap,
            "max_motion_gap_m": round(self.max_motion_gap_m, 2),
            "worst_motion_gap": self.worst_motion_gap,
            "max_cluster_size": self.max_cluster_size,
            "largest_cluster": self.largest_cluster,
            "n_selected": self.n_selected,
            "n_cells_mission": self.coverage.n_cells_mission,
            "n_cells_covered": self.coverage.n_cells_covered,
            "n_cells_at_target": self.coverage.n_cells_at_target,
            "pct_covered_at_target": round(self.coverage.pct_covered_at_target, 4),
            "pct_mission_at_target": round(self.coverage.pct_mission_at_target, 4),
            "mean_views_per_covered_cell": round(self.coverage.mean_views_per_covered_cell, 2),
        }
        if self.parallax is not None:
            out["parallax"] = self.parallax.as_dict()
        return out


class SelectionFailed(Exception):
    def __init__(self, health: SelectionHealth) -> None:
        self.health = health
        super().__init__("; ".join(health.failures))


def assess_selection(
    candidates: pd.DataFrame,
    params: SelectionParams,
    *,
    view_counts: ViewCounts | None = None,
    mission_cells: frozenset[tuple[int, int]] | None = None,
    footprints: FootprintCache | None = None,
    grid: GroundGrid | None = None,
    cell_ground_z: CellGroundZ | None = None,
) -> SelectionHealth:
    selected = candidates[candidates["selected"]].sort_index()
    failures: list[str] = []
    n_selected = len(selected)

    max_temporal_gap = 0
    worst_temporal_gap = None
    max_motion_gap_m = 0.0
    worst_motion_gap: dict | None = None
    for gap in iter_selection_gaps(selected):
        if gap.gap_frames > max_temporal_gap:
            max_temporal_gap = gap.gap_frames
            worst_temporal_gap = gap.as_dict()
        if gap.gap_m > max_motion_gap_m:
            max_motion_gap_m = gap.gap_m
            worst_motion_gap = gap.as_dict()

    if n_selected == 0:
        failures.append("no frames selected")
    elif max_motion_gap_m > params.max_motion_gap_m:
        log.warning(
            "max motion gap %.1fm exceeds limit %.1fm (warn only; ODM matches globally)",
            max_motion_gap_m,
            params.max_motion_gap_m,
        )

    max_cluster_size, densest_members = max_local_density(
        [int(i) for i in selected.index], candidates, params
    )
    largest_cluster = (
        {"size": max_cluster_size, "frame_numbers": densest_members[:10]}
        if max_cluster_size
        else None
    )

    coverage = coverage_metrics(
        view_counts or {},
        params.target_views_per_cell,
        mission_cells=mission_cells,
        bin_size_m=params.bin_size_m,
    )

    # Honest hard gate: did selection actually cover the site? Only enforced when we
    # have a real mission region to measure against (footprints/view_counts present).
    if n_selected and view_counts and coverage.n_cells_mission:
        pct_covered = coverage.n_cells_covered / coverage.n_cells_mission
        if pct_covered < params.min_pct_mission_covered:
            failures.append(
                f"only {pct_covered:.0%} of mission cells covered "
                f"(need {params.min_pct_mission_covered:.0%}); budget likely too small"
            )

    # Convergence/parallax is a verification metric (reported, never a gate): ODM
    # triangulates from whatever baselines the even sampling produced.
    parallax: ParallaxMetrics | None = None
    if (
        n_selected
        and footprints is not None
        and grid is not None
        and candidates["selected"].any()
    ):
        from scene_recon.selection.parallax import cell_viewers_from_selection

        ground_z = cell_ground_z
        if ground_z is None:
            cells = mission_cells or grid.mission_cells(footprints.values())
            ground_z = approx_cell_ground_z(grid, cells, candidates)
        ctx = build_parallax_context(candidates)
        viewers = cell_viewers_from_selection(candidates, footprints)
        parallax = parallax_metrics(viewers, ctx, grid, ground_z, params)

    return SelectionHealth(
        passed=not failures,
        failures=failures,
        max_temporal_gap=max_temporal_gap,
        worst_temporal_gap=worst_temporal_gap,
        max_motion_gap_m=max_motion_gap_m,
        worst_motion_gap=worst_motion_gap,
        max_cluster_size=max_cluster_size,
        largest_cluster=largest_cluster,
        n_selected=n_selected,
        coverage=coverage,
        parallax=parallax,
    )
