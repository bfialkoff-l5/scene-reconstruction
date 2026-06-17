from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

SELECTION_POLICY = "path_walk_v2"

BIN_SIZE_M = 5.0
MIN_ALTITUDE_M = 0.0
MIN_TRANSLATION_M = 2.0
MIN_ROTATION_DEG = 5.0
MAX_FRAME_GAP = 90
CLUSTER_RADIUS_M = 10.0
MAX_PER_CLUSTER = 3
COVERAGE_WARN_M = 15.0
MAX_KEYFRAMES = 500


@dataclass(frozen=True)
class SelectionParams:
    bin_size_m: float = BIN_SIZE_M
    min_altitude_m: float = MIN_ALTITUDE_M
    min_translation_m: float = MIN_TRANSLATION_M
    min_rotation_deg: float = MIN_ROTATION_DEG
    max_frame_gap: int = MAX_FRAME_GAP
    cluster_radius_m: float = CLUSTER_RADIUS_M
    max_per_cluster: int = MAX_PER_CLUSTER
    coverage_warn_m: float = COVERAGE_WARN_M
    max_keyframes: int = MAX_KEYFRAMES

    def as_constants(self) -> dict:
        return {
            "policy": SELECTION_POLICY,
            **asdict(self),
        }


DEFAULT_SELECTION_PARAMS = SelectionParams()


def _wrap_angle(rad: float) -> float:
    return (rad + math.pi) % (2 * math.pi) - math.pi


def translation_m(a: pd.Series, b: pd.Series) -> float:
    de = float(a["easting"] - b["easting"])
    dn = float(a["northing"] - b["northing"])
    return math.hypot(de, dn)


def rotation_deg(a: pd.Series, b: pd.Series) -> float:
    dr = _wrap_angle(float(a["roll_rad"] - b["roll_rad"]))
    dp = _wrap_angle(float(a["pitch_rad"] - b["pitch_rad"]))
    dy = _wrap_angle(float(a["yaw_rad"] - b["yaw_rad"]))
    return math.degrees(math.sqrt(dr * dr + dp * dp + dy * dy))


def assign_bins(candidates: pd.DataFrame, params: SelectionParams) -> pd.DataFrame:
    out = candidates.copy()
    origin_e = out["easting"].min()
    origin_n = out["northing"].min()
    out["cell_x"] = np.floor((out["easting"] - origin_e) / params.bin_size_m).astype("Int64")
    out["cell_y"] = np.floor((out["northing"] - origin_n) / params.bin_size_m).astype("Int64")
    return out


def _bridge_after(pool: pd.DataFrame, start: int, max_gap: int) -> int:
    end = start + max_gap
    window = pool.loc[start + 1 : end]
    if window.empty:
        return start
    return int(window.index[-1])


def _best_in_window(pool: pd.DataFrame, start: int, end: int) -> int:
    return int(pool.loc[start:end]["quality_score"].idxmax())


def _best_in_bin_window(
    pool: pd.DataFrame,
    out: pd.DataFrame,
    start: int,
    end: int,
    cell: tuple,
) -> int | None:
    window = pool.loc[start:end]
    cx, cy = cell
    in_bin = window[(window["cell_x"] == cx) & (window["cell_y"] == cy)]
    if in_bin.empty:
        return None
    return int(in_bin["quality_score"].idxmax())


def _path_walk(pool: pd.DataFrame, out: pd.DataFrame, params: SelectionParams) -> list[int]:
    frames = [int(i) for i in pool.sort_index().index]
    if not frames:
        return []

    picked: list[int] = []
    segment_start = frames[0]
    segment_end = segment_start
    for frame in frames:
        if frame - segment_start > params.max_frame_gap:
            break
        segment_end = frame

    picked.append(_best_in_window(pool, segment_start, segment_end))

    for frame in frames:
        if frame <= picked[-1]:
            continue

        last = picked[-1]
        while frame - last > params.max_frame_gap:
            bridge = _bridge_after(pool, last, params.max_frame_gap)
            if bridge <= last:
                break
            picked.append(bridge)
            last = bridge

        row = out.loc[frame]
        last_row = out.loc[last]
        cell = (int(row["cell_x"]), int(row["cell_y"]))
        last_cell = (int(last_row["cell_x"]), int(last_row["cell_y"]))
        if cell == last_cell:
            continue
        if translation_m(row, last_row) < params.min_translation_m:
            continue

        best = _best_in_bin_window(pool, out, last, frame, cell)
        if best is not None and best not in picked:
            picked.append(best)

    return picked


def _same_cluster(
    idx: int,
    rep: int,
    row: pd.Series,
    rep_row: pd.Series,
    params: SelectionParams,
) -> bool:
    if abs(idx - rep) > params.max_frame_gap:
        return False
    return translation_m(row, rep_row) <= params.cluster_radius_m


def group_local_clusters(indices: list[int], out: pd.DataFrame, params: SelectionParams) -> list[list[int]]:
    clusters: list[list[int]] = []
    for idx in sorted(indices):
        row = out.loc[idx]
        placed = False
        for cluster in clusters:
            rep = cluster[0]
            if _same_cluster(idx, rep, row, out.loc[rep], params):
                cluster.append(idx)
                placed = True
                break
        if not placed:
            clusters.append([idx])
    return clusters


def _cap_clusters(
    picked: list[int],
    out: pd.DataFrame,
    params: SelectionParams,
) -> tuple[set[int], set[int]]:
    kept: set[int] = set()
    capped: set[int] = set()
    clusters = group_local_clusters(picked, out, params)

    for cluster in clusters:
        if len(cluster) <= params.max_per_cluster:
            kept.update(cluster)
            continue

        cluster = sorted(cluster)
        stride = len(cluster) / params.max_per_cluster
        chosen = {cluster[int(i * stride)] for i in range(params.max_per_cluster)}
        kept.update(chosen)
        capped.update(idx for idx in cluster if idx not in chosen)

    return kept, capped


def _apply_rotation_diversity(
    kept: set[int],
    out: pd.DataFrame,
    params: SelectionParams,
) -> tuple[set[int], set[int]]:
    final = set(kept)
    dropped: set[int] = set()
    clusters = group_local_clusters(list(kept), out, params)

    for cluster in clusters:
        ranked = sorted(
            cluster,
            key=lambda i: float(out.loc[i, "quality_score"]),
            reverse=True,
        )
        cluster_kept = {ranked[0]}
        for idx in ranked[1:]:
            row = out.loc[idx]
            min_trans = min(translation_m(row, out.loc[a]) for a in cluster_kept)
            min_rot = min(rotation_deg(row, out.loc[a]) for a in cluster_kept)
            if min_trans >= params.min_translation_m or min_rot >= params.min_rotation_deg:
                cluster_kept.add(idx)
            else:
                final.discard(idx)
                dropped.add(idx)

    return final, dropped


def _finalize_temporal_chain(
    kept: set[int],
    pool: pd.DataFrame,
    params: SelectionParams,
) -> set[int]:
    ordered = sorted(kept)
    if not ordered:
        return kept

    chain = [ordered[0]]
    for idx in ordered[1:]:
        while idx - chain[-1] > params.max_frame_gap:
            bridge = _bridge_after(pool, chain[-1], params.max_frame_gap)
            if bridge <= chain[-1]:
                break
            chain.append(bridge)
        if idx not in chain:
            chain.append(idx)
    return set(chain)


def _stabilize_selection(
    kept: set[int],
    pool: pd.DataFrame,
    out: pd.DataFrame,
    params: SelectionParams,
) -> tuple[set[int], set[int]]:
    cluster_capped: set[int] = set()
    for _ in range(16):
        before = frozenset(kept)
        kept = _finalize_temporal_chain(kept, pool, params)
        kept, capped = _cap_clusters(sorted(kept), out, params)
        cluster_capped |= capped
        if frozenset(kept) == before:
            break
    return kept, cluster_capped


def select_keyframes(
    candidates: pd.DataFrame,
    params: SelectionParams | None = None,
) -> pd.DataFrame:
    """Select keyframes from a fully scored candidate table."""
    p = params or DEFAULT_SELECTION_PARAMS
    out = assign_bins(candidates, p)
    out["selected"] = False
    out["reject_reason"] = pd.NA

    eligible = out["altamsl"] >= p.min_altitude_m
    out.loc[~eligible, "reject_reason"] = "below_altitude"

    pool = out.loc[eligible & out["quality_score"].notna()]
    if pool.empty:
        return out

    path_picked = _path_walk(pool, out, p)
    kept, cluster_capped = _stabilize_selection(set(path_picked), pool, out, p)
    kept, diversity_dropped = _apply_rotation_diversity(kept, out, p)
    kept, more_capped = _stabilize_selection(kept, pool, out, p)
    cluster_capped |= more_capped

    if len(kept) > p.max_keyframes:
        ranked = sorted(
            kept,
            key=lambda i: float(out.loc[i, "quality_score"]),
            reverse=True,
        )
        kept = set(ranked[: p.max_keyframes])
        keyframe_dropped = set(ranked[p.max_keyframes :])
    else:
        keyframe_dropped = set()

    kept, final_capped = _stabilize_selection(kept, pool, out, p)
    cluster_capped |= final_capped

    out.loc[list(kept), "selected"] = True

    path_set = set(path_picked)
    for idx in pool.index:
        idx = int(idx)
        if out.loc[idx, "selected"]:
            continue
        if idx in cluster_capped:
            out.loc[idx, "reject_reason"] = "spatial_cluster_cap"
        elif idx in diversity_dropped:
            out.loc[idx, "reject_reason"] = "low_pose_novelty"
        elif idx in keyframe_dropped:
            out.loc[idx, "reject_reason"] = "max_keyframes"
        elif idx in path_set:
            out.loc[idx, "reject_reason"] = "not_retained_after_cap"
        else:
            out.loc[idx, "reject_reason"] = "not_on_path"

    return out
