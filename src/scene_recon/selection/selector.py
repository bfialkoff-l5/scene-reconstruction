from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from scene_recon.selection.cluster import BallIndex, filter_to_main_component
from scene_recon.selection.footprint import (
    FootprintCache,
    ViewCounts,
    assign_bins,
    infer_ground_altitude_m,
    recount_views,
)
from scene_recon.selection.metrics import (
    CandidateMetrics,
    coverage_selects,
    score_candidate,
    write_debug_columns,
)
from scene_recon.selection.params import DEFAULT_SELECTION_PARAMS, SelectionParams


@dataclass
class SelectionState:
    out: pd.DataFrame
    params: SelectionParams
    ground_altitude_m: float
    eastings: np.ndarray
    northings: np.ndarray
    index_pos: dict[int, int]
    view_counts: ViewCounts
    footprint_cache: FootprintCache
    ball_size: dict[int, int]
    selected: set[int]
    selected_order: list[int]
    ball_index: BallIndex


def _init_debug_columns(out: pd.DataFrame) -> pd.DataFrame:
    out["selected"] = False
    out["reject_reason"] = pd.NA
    out["selection_reason"] = pd.NA
    out["footprint_agl_m"] = pd.NA
    out["footprint_area_m2"] = pd.NA
    out["footprint_cell_count"] = pd.NA
    out["coverage_gain_cells"] = pd.NA
    out["coverage_gain_ratio"] = pd.NA
    out["distance_from_last_selected_m"] = pd.NA
    out["rotation_from_last_selected_deg"] = pd.NA
    out["selector_score"] = pd.NA
    return out


def _frame_windows(frames: list[int], max_frame_gap: int) -> list[tuple[int, int]]:
    if not frames:
        return []
    windows: list[tuple[int, int]] = []
    start = frames[0]
    last = frames[-1]
    while start <= last:
        windows.append((start, start + max_frame_gap - 1))
        start += max_frame_gap
    return windows


def _nearest_selected_before(idx: int, selected_order: list[int]) -> int | None:
    ordered = sorted(selected_order)
    pos = bisect.bisect_left(ordered, idx)
    if pos == 0:
        return None
    return ordered[pos - 1]


def _best_in_window(
    state: SelectionState,
    pool: pd.DataFrame,
    start: int,
    end: int,
    *,
    last_selected: int | None,
    require_coverage: bool,
) -> tuple[int, CandidateMetrics] | None:
    window = pool.loc[start:end]
    if window.empty:
        return None

    best_idx: int | None = None
    best_metrics: CandidateMetrics | None = None
    best_score = -math.inf

    for idx_obj in window.index:
        idx = int(idx_obj)
        if idx in state.selected:
            continue
        if state.ball_index.would_exceed_cap(
            idx, state.selected, state.params.max_per_cluster, state.ball_size
        ):
            continue

        resolved_last = last_selected
        if last_selected is None:
            resolved_last = _nearest_selected_before(idx, state.selected_order)

        metrics = score_candidate(
            idx,
            state.out,
            state.params,
            ground_altitude_m=state.ground_altitude_m,
            view_counts=state.view_counts,
            last_selected=resolved_last,
            footprint_cache=state.footprint_cache,
        )
        if require_coverage and not coverage_selects(metrics, state.params):
            continue
        if metrics.score > best_score:
            best_idx = idx
            best_metrics = metrics
            best_score = metrics.score

    if best_idx is None or best_metrics is None:
        return None
    return best_idx, best_metrics


def _select_frame(
    state: SelectionState,
    idx: int,
    reason: str,
    metrics: CandidateMetrics,
) -> None:
    state.ball_index.record_selection(idx, state.selected, state.ball_size)
    state.selected.add(idx)
    state.selected_order.append(idx)
    for c in metrics.cells:
        state.view_counts[c] = state.view_counts.get(c, 0) + 1
    write_debug_columns(state.out, idx, metrics)
    state.out.loc[idx, "selected"] = True
    state.out.loc[idx, "selection_reason"] = reason
    state.out.loc[idx, "reject_reason"] = pd.NA


def _reject_reason(
    idx: int,
    state: SelectionState,
    metrics: CandidateMetrics,
) -> str:
    params = state.params
    is_saturated = metrics.gain <= 0
    is_marginal = (
        not is_saturated
        and metrics.gain_ratio < params.min_coverage_gain_ratio
        and metrics.gain < params.min_coverage_gain_cells
    )
    has_no_motion = (
        metrics.distance_m < params.min_translation_m
        and metrics.rotation_deg < params.min_rotation_deg
    )
    is_dense = state.ball_index.would_exceed_cap(
        idx, state.selected, params.max_per_cluster, state.ball_size
    )

    if is_saturated:
        return "coverage_saturated"
    if is_marginal:
        return "low_coverage_gain"
    if has_no_motion:
        return "low_pose_novelty"
    if is_dense:
        return "spatial_cluster_cap"
    if len(state.selected) >= params.max_keyframes:
        return "max_keyframes"
    return "not_selected"


def _select_keyframes_inner(
    out: pd.DataFrame,
    pool: pd.DataFrame,
    params: SelectionParams,
) -> pd.DataFrame:
    ground_altitude_m = infer_ground_altitude_m(pool)
    frames = [int(i) for i in pool.index]
    eastings = out["easting"].astype(float).to_numpy()
    northings = out["northing"].astype(float).to_numpy()
    index_pos = {int(idx): i for i, idx in enumerate(out.index)}
    ball_index = BallIndex(eastings, northings, index_pos, params.cluster_radius_m)

    state = SelectionState(
        out=out,
        params=params,
        ground_altitude_m=ground_altitude_m,
        eastings=eastings,
        northings=northings,
        index_pos=index_pos,
        view_counts={},
        footprint_cache={},
        ball_size={},
        selected=set(),
        selected_order=[],
        ball_index=ball_index,
    )

    first = frames[0]
    last_frame = frames[-1]
    seed_end = first + params.max_frame_gap
    seed = _best_in_window(
        state,
        pool,
        first,
        seed_end,
        last_selected=None,
        require_coverage=False,
    )
    if seed is None:
        return out
    seed_idx, seed_metrics = seed
    _select_frame(state, seed_idx, "temporal_seed", seed_metrics)

    chain_cursor = state.selected_order[-1]
    while chain_cursor < last_frame and len(state.selected) < params.max_keyframes:
        last_selected = state.selected_order[-1]
        bridge = _best_in_window(
            state,
            pool,
            chain_cursor + 1,
            chain_cursor + params.max_frame_gap,
            last_selected=last_selected,
            require_coverage=False,
        )
        if bridge is None:
            chain_cursor += params.max_frame_gap
            continue
        bridge_idx, bridge_metrics = bridge
        _select_frame(state, bridge_idx, "temporal_chain", bridge_metrics)
        chain_cursor = bridge_idx

    windows = _frame_windows(frames, params.max_frame_gap)
    made_progress = True
    while len(state.selected) < params.max_keyframes and made_progress:
        made_progress = False
        for start, end in windows:
            if len(state.selected) >= params.max_keyframes:
                break
            fill = _best_in_window(
                state,
                pool,
                start,
                end,
                last_selected=None,
                require_coverage=True,
            )
            if fill is None:
                continue
            idx, metrics = fill
            _select_frame(state, idx, "coverage_gain", metrics)
            made_progress = True

    made_progress = True
    while len(state.selected) < params.max_keyframes and made_progress:
        made_progress = False
        for start, end in windows:
            if len(state.selected) >= params.max_keyframes:
                break
            fill = _best_in_window(
                state,
                pool,
                start,
                end,
                last_selected=None,
                require_coverage=False,
            )
            if fill is None:
                continue
            idx, metrics = fill
            _select_frame(state, idx, "budget_fill", metrics)
            made_progress = True

    keep = filter_to_main_component(state.selected, out, params)
    dropped = state.selected - keep
    if dropped:
        for idx in dropped:
            out.loc[idx, "selected"] = False
            out.loc[idx, "selection_reason"] = pd.NA
            out.loc[idx, "reject_reason"] = "outlier_component"
        state.view_counts = recount_views(
            keep, out, params, ground_altitude_m, state.footprint_cache
        )
        state.ball_size = ball_index.rebuild_ball_sizes(keep)
        state.selected = keep
        state.selected_order = [i for i in state.selected_order if i in keep]

    for idx_obj in pool.index:
        idx = int(idx_obj)
        if idx not in state.selected and pd.isna(out.loc[idx, "reject_reason"]):
            last_selected = _nearest_selected_before(idx, state.selected_order)
            metrics = score_candidate(
                idx,
                out,
                params,
                ground_altitude_m=ground_altitude_m,
                view_counts=state.view_counts,
                last_selected=last_selected,
                footprint_cache=state.footprint_cache,
            )
            out.loc[idx, "reject_reason"] = _reject_reason(idx, state, metrics)

    return out


def select_keyframes(
    candidates: pd.DataFrame,
    params: SelectionParams | None = None,
) -> pd.DataFrame:
    p = params or DEFAULT_SELECTION_PARAMS
    out = assign_bins(candidates, p)
    out = _init_debug_columns(out)

    eligible = out["altamsl"] >= p.min_altitude_m
    out.loc[~eligible, "reject_reason"] = "below_altitude"
    scored = (
        out["quality_score"].notna()
        if "quality_score" in out.columns
        else pd.Series(False, index=out.index)
    )
    out.loc[eligible & ~scored, "reject_reason"] = "missing_quality_score"

    pool = out.loc[eligible & scored].sort_index()
    if pool.empty:
        return out

    return _select_keyframes_inner(out, pool, p)
