from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd

from scene_recon.selection.footprint import (
    FootprintCache,
    ViewCounts,
    footprint_cells,
    footprint_for_row,
    rotation_deg,
    translation_m,
)
from scene_recon.selection.params import SelectionParams


@dataclass(frozen=True)
class CandidateMetrics:
    cells: frozenset[tuple[int, int]]
    gain: int
    gain_ratio: float
    distance_m: float
    rotation_deg: float
    score: float
    agl_m: float
    area_m2: float
    cell_count: int


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
    n_cells_covered: int
    n_cells_at_target: int
    pct_cells_at_target: float
    mean_views_per_cell: float
    max_views_per_cell: int
    views_histogram: dict[str, int]

    def as_dict(self) -> dict:
        return {
            "n_cells_covered": self.n_cells_covered,
            "n_cells_at_target": self.n_cells_at_target,
            "pct_cells_at_target": round(self.pct_cells_at_target, 4),
            "mean_views_per_cell": round(self.mean_views_per_cell, 2),
            "max_views_per_cell": self.max_views_per_cell,
            "views_histogram": self.views_histogram,
        }


def coverage_metrics(view_counts: ViewCounts, target: int) -> CoverageMetrics:
    if not view_counts:
        return CoverageMetrics(0, 0, 0.0, 0.0, 0, {})
    n_cells = len(view_counts)
    histogram = Counter(view_counts.values())
    n_at_target = sum(1 for v in view_counts.values() if v >= target)
    return CoverageMetrics(
        n_cells_covered=n_cells,
        n_cells_at_target=n_at_target,
        pct_cells_at_target=n_at_target / n_cells,
        mean_views_per_cell=sum(view_counts.values()) / n_cells,
        max_views_per_cell=max(view_counts.values()),
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


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out):
        return default
    return out


def score_candidate(
    idx: int,
    out: pd.DataFrame,
    params: SelectionParams,
    *,
    ground_altitude_m: float,
    view_counts: ViewCounts,
    last_selected: int | None,
    footprint_cache: FootprintCache | None = None,
) -> CandidateMetrics:
    row = out.loc[idx]
    if footprint_cache is not None and idx in footprint_cache:
        cells, agl_m, area_m2, cell_count = footprint_cache[idx]
    else:
        footprint = footprint_for_row(row, params, ground_altitude_m=ground_altitude_m)
        cells = footprint_cells(footprint, cell_size_m=params.bin_size_m)
        agl_m = footprint.agl_m
        area_m2 = footprint.area_m2
        cell_count = len(cells)
        if footprint_cache is not None:
            footprint_cache[idx] = (cells, agl_m, area_m2, cell_count)

    target = params.target_views_per_cell
    gain = sum(1 for c in cells if view_counts.get(c, 0) < target)
    gain_ratio = gain / max(len(cells), 1)

    if last_selected is None:
        distance = math.inf
        rotation = math.inf
    else:
        distance = translation_m(row, out.loc[last_selected])
        rotation = rotation_deg(row, out.loc[last_selected])

    quality = _safe_float(row.get("quality_score", 0.0))
    trans_score = min(distance / max(params.min_translation_m, 1e-6), 1.0)
    rot_score = min(rotation / max(params.min_rotation_deg, 1e-6), 1.0)
    novelty = max(trans_score, rot_score)
    score = (
        params.score_quality_weight * quality
        + params.score_coverage_weight * gain_ratio
        + params.score_novelty_weight * novelty
    )

    return CandidateMetrics(
        cells=frozenset(cells),
        gain=gain,
        gain_ratio=gain_ratio,
        distance_m=distance,
        rotation_deg=rotation,
        score=score,
        agl_m=agl_m,
        area_m2=area_m2,
        cell_count=cell_count,
    )


def write_debug_columns(out: pd.DataFrame, idx: int, metrics: CandidateMetrics) -> None:
    out.loc[idx, "footprint_agl_m"] = metrics.agl_m
    out.loc[idx, "footprint_area_m2"] = metrics.area_m2
    out.loc[idx, "footprint_cell_count"] = metrics.cell_count
    out.loc[idx, "coverage_gain_cells"] = metrics.gain
    out.loc[idx, "coverage_gain_ratio"] = metrics.gain_ratio
    out.loc[idx, "distance_from_last_selected_m"] = (
        metrics.distance_m if math.isfinite(metrics.distance_m) else pd.NA
    )
    out.loc[idx, "rotation_from_last_selected_deg"] = (
        metrics.rotation_deg if math.isfinite(metrics.rotation_deg) else pd.NA
    )
    out.loc[idx, "selector_score"] = metrics.score


def coverage_selects(metrics: CandidateMetrics, params: SelectionParams) -> bool:
    if metrics.gain <= 0:
        return False
    if metrics.gain_ratio >= params.min_coverage_gain_ratio:
        return True
    if metrics.gain >= params.min_coverage_gain_cells:
        return (
            metrics.distance_m >= params.min_translation_m
            or metrics.rotation_deg >= params.min_rotation_deg
        )
    return False
