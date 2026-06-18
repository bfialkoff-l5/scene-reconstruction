from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd

from scene_recon.selection.footprint import ViewCounts, translation_m


@dataclass(frozen=True)
class GapInfo:
    from_frame: int
    to_frame: int
    gap_frames: int
    gap_m: float

    def as_dict(self) -> dict:
        return {
            "from_frame": self.from_frame,
            "to_frame": self.to_frame,
            "gap_frames": self.gap_frames,
            "gap_m": round(self.gap_m, 2),
        }


@dataclass(frozen=True)
class CoverageMetrics:
    n_cells_mission: int
    n_cells_covered: int
    n_cells_at_target: int
    pct_covered_at_target: float
    pct_mission_at_target: float
    mean_views_per_covered_cell: float
    max_views_per_cell: int
    n_cells_over_2x_target: int
    over_coverage_area_m2: float
    views_histogram: dict[str, int]

    def as_dict(self) -> dict:
        return {
            "n_cells_mission": self.n_cells_mission,
            "n_cells_covered": self.n_cells_covered,
            "n_cells_at_target": self.n_cells_at_target,
            "pct_covered_at_target": round(self.pct_covered_at_target, 4),
            "pct_mission_at_target": round(self.pct_mission_at_target, 4),
            "mean_views_per_covered_cell": round(self.mean_views_per_covered_cell, 2),
            "max_views_per_cell": self.max_views_per_cell,
            "n_cells_over_2x_target": self.n_cells_over_2x_target,
            "over_coverage_area_m2": round(self.over_coverage_area_m2, 1),
            "views_histogram": self.views_histogram,
        }


def coverage_metrics(
    view_counts: ViewCounts,
    target: int,
    *,
    mission_cells: frozenset[tuple[int, int]] | None = None,
    bin_size_m: float = 5.0,
) -> CoverageMetrics:
    n_mission = len(mission_cells) if mission_cells is not None else len(view_counts)
    if not view_counts:
        return CoverageMetrics(n_mission, 0, 0, 0.0, 0.0, 0.0, 0, 0, 0.0, {})
    n_covered = len(view_counts)
    histogram = Counter(view_counts.values())
    n_at_target = sum(1 for v in view_counts.values() if v >= target)
    n_over = sum(1 for v in view_counts.values() if v >= 2 * target)
    return CoverageMetrics(
        n_cells_mission=n_mission,
        n_cells_covered=n_covered,
        n_cells_at_target=n_at_target,
        pct_covered_at_target=n_at_target / n_covered,
        pct_mission_at_target=n_at_target / max(n_mission, 1),
        mean_views_per_covered_cell=sum(view_counts.values()) / n_covered,
        max_views_per_cell=max(view_counts.values()),
        n_cells_over_2x_target=n_over,
        over_coverage_area_m2=n_over * bin_size_m * bin_size_m,
        views_histogram={str(k): int(histogram[k]) for k in sorted(histogram)},
    )


def iter_selection_gaps(selected: pd.DataFrame) -> Iterator[GapInfo]:
    if len(selected) < 2:
        return
    sel = selected.sort_index()
    prev_idx: int | None = None
    for idx, row in sel.iterrows():
        idx = int(idx)
        if prev_idx is not None:
            yield GapInfo(
                from_frame=prev_idx,
                to_frame=idx,
                gap_frames=idx - prev_idx,
                gap_m=translation_m(row, sel.loc[prev_idx]),
            )
        prev_idx = idx
