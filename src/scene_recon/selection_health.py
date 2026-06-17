from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from scene_recon.selection import (
    SelectionParams,
    compute_view_counts,
    coverage_metrics,
    iter_selection_gaps,
    max_local_density,
)
from scene_recon.selection.footprint import ViewCounts
from scene_recon.selection.metrics import CoverageMetrics

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

    @property
    def n_cells_covered(self) -> int:
        return self.coverage.n_cells_covered

    @property
    def n_cells_at_target(self) -> int:
        return self.coverage.n_cells_at_target

    @property
    def pct_cells_at_target(self) -> float:
        return self.coverage.pct_cells_at_target

    @property
    def mean_views_per_cell(self) -> float:
        return self.coverage.mean_views_per_cell

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "failures": self.failures,
            "max_temporal_gap": self.max_temporal_gap,
            "worst_temporal_gap": self.worst_temporal_gap,
            "max_motion_gap_m": round(self.max_motion_gap_m, 2),
            "worst_motion_gap": self.worst_motion_gap,
            "max_cluster_size": self.max_cluster_size,
            "largest_cluster": self.largest_cluster,
            "n_selected": self.n_selected,
            "n_cells_covered": self.coverage.n_cells_covered,
            "n_cells_at_target": self.coverage.n_cells_at_target,
            "pct_cells_at_target": round(self.coverage.pct_cells_at_target, 4),
            "mean_views_per_cell": round(self.coverage.mean_views_per_cell, 2),
        }


class SelectionFailed(Exception):
    def __init__(self, health: SelectionHealth) -> None:
        self.health = health
        super().__init__("; ".join(health.failures))


def assess_selection(
    candidates: pd.DataFrame,
    params: SelectionParams,
    *,
    view_counts: ViewCounts | None = None,
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
        failures.append(
            f"max motion gap {max_motion_gap_m:.1f}m exceeds limit {params.max_motion_gap_m:.1f}m"
        )

    if n_selected >= 2 and max_temporal_gap > params.max_frame_gap:
        log.warning(
            "temporal gap %d frames exceeds max_frame_gap %d (warn only)",
            max_temporal_gap,
            params.max_frame_gap,
        )

    max_cluster_size, densest_members = max_local_density(
        [int(i) for i in selected.index], candidates, params
    )
    largest_cluster = (
        {"size": max_cluster_size, "frame_numbers": densest_members[:10]}
        if max_cluster_size
        else None
    )

    if max_cluster_size > params.max_per_cluster:
        failures.append(
            f"spatial cluster size {max_cluster_size} exceeds limit {params.max_per_cluster}"
        )

    if view_counts is None:
        view_counts = compute_view_counts(candidates, params) if n_selected else {}
    coverage = coverage_metrics(view_counts, params.target_views_per_cell)

    if n_selected and coverage.pct_cells_at_target < params.min_pct_cells_at_target:
        failures.append(
            f"only {coverage.pct_cells_at_target:.0%} of {coverage.n_cells_covered} covered cells reach "
            f"target {params.target_views_per_cell} views (need {params.min_pct_cells_at_target:.0%})"
        )

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
    )
