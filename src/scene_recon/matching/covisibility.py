"""Geometric co-visibility graph: which keyframe pairs actually see the same ground.

Two frames are co-visible when their ground footprints (the terrain cells their frustum
hits, computed once by the selection ray-march) overlap *and* they look at it from a
similar enough direction to be matchable by SIFT. This is the per-flight, trajectory-
agnostic replacement for a fixed GPS-neighbour count: an orbit, a lawnmower grid and a
single pass each produce the candidate set they need, with no magic constant.

Pure numpy (no scipy): O(n^2) over keyframes (hundreds), trivially fast.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

Cell = tuple[int, int]
CellSet = frozenset[Cell]


@dataclass(frozen=True)
class CoVisEdge:
    """A predicted matchable pair, scored by geometry alone."""

    i: int  # frame number
    j: int  # frame number
    iom: float  # |cells_i & cells_j| / min(|cells_i|, |cells_j|)
    shared_cells: int
    centroid_dist_m: float
    view_angle_deg: float  # angle between the two optical axes (nan if unknown)


@dataclass(frozen=True)
class CoVisGraph:
    """Predicted co-visibility over a set of keyframes.

    `frames` is kept sorted so that the index of a frame in this list is its along-track
    rank -- the cheap proxy that separates *sequential* edges (small rank gap) from the
    *cross-track / loop-closure* edges (large rank gap) the matcher must not miss.
    """

    frames: list[int]
    edges: list[CoVisEdge]

    def _rank(self) -> dict[int, int]:
        return {f: r for r, f in enumerate(self.frames)}

    def degrees(self) -> dict[int, int]:
        deg = {f: 0 for f in self.frames}
        for e in self.edges:
            deg[e.i] += 1
            deg[e.j] += 1
        return deg

    def summary(self, *, cross_track_rank_gap: int = 16) -> dict:
        """Topology stats that diagnose the matching profile."""
        rank = self._rank()
        deg = np.array(list(self.degrees().values()), dtype=float)
        dists = np.array([e.centroid_dist_m for e in self.edges], dtype=float)
        gaps = np.array(
            [abs(rank[e.i] - rank[e.j]) for e in self.edges], dtype=int
        )
        cross = int((gaps > cross_track_rank_gap).sum())
        return {
            "n_frames": len(self.frames),
            "n_edges": len(self.edges),
            "cross_track_edges": cross,
            "cross_track_frac": float(cross / len(self.edges)) if self.edges else 0.0,
            "degree_p50": float(np.percentile(deg, 50)) if deg.size else 0.0,
            "degree_p95": float(np.percentile(deg, 95)) if deg.size else 0.0,
            "degree_max": float(deg.max()) if deg.size else 0.0,
            "reach_p50_m": float(np.percentile(dists, 50)) if dists.size else 0.0,
            "reach_p95_m": float(np.percentile(dists, 95)) if dists.size else 0.0,
            "reach_max_m": float(dists.max()) if dists.size else 0.0,
        }


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    na = a / (np.linalg.norm(a) + 1e-12)
    nb = b / (np.linalg.norm(b) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(na @ nb, -1.0, 1.0))))


def build_covisibility(
    cells_by_frame: Mapping[int, CellSet],
    positions: Mapping[int, tuple[float, float]],
    *,
    view_dirs: Mapping[int, np.ndarray] | None = None,
    iom_min: float = 0.10,
    view_angle_max_deg: float = 45.0,
    pair_budget: int | None = None,
) -> CoVisGraph:
    """Build the co-visibility graph from precomputed footprint cell-sets.

    Args:
        cells_by_frame: frame number -> set of terrain cells the frame sees.
        positions: frame number -> (easting, northing) for the edge baseline length.
        view_dirs: frame number -> optical-axis unit vector. When given, pairs whose
            axes diverge past `view_angle_max_deg` are dropped (oblique frames looking at
            the same patch from opposite sides do not match in SIFT). When omitted, the
            angle gate is skipped and `view_angle_deg` is recorded as nan.
        iom_min: minimum intersection-over-min overlap to keep a pair.
        pair_budget: if set, keep only each frame's top-`pair_budget` edges by `iom`
            (caps matching cost as flights grow; the union is symmetric so a frame may
            still exceed the budget via a neighbour that kept it).

    Returns:
        CoVisGraph with frames sorted ascending (frame number == along-track order).
    """
    frames = sorted(f for f in cells_by_frame if cells_by_frame[f])
    n = len(frames)
    edges: list[CoVisEdge] = []
    # Direct O(n^2) set intersection: correct and fast for hundreds of frames.
    for a in range(n):
        fa = frames[a]
        ca = cells_by_frame[fa]
        la = len(ca)
        pa = positions[fa]
        va = None if view_dirs is None else view_dirs.get(fa)
        for b in range(a + 1, n):
            fb = frames[b]
            cb = cells_by_frame[fb]
            shared = len(ca & cb)
            if shared == 0:
                continue
            iom = shared / min(la, len(cb))
            if iom < iom_min:
                continue
            ang = float("nan")
            if view_dirs is not None and va is not None:
                vb = view_dirs.get(fb)
                if vb is not None:
                    ang = _angle_between(va, vb)
                    if ang > view_angle_max_deg:
                        continue
            pb = positions[fb]
            dist = float(np.hypot(pa[0] - pb[0], pa[1] - pb[1]))
            edges.append(CoVisEdge(fa, fb, iom, shared, dist, ang))

    if pair_budget is not None:
        edges = _apply_pair_budget(frames, edges, pair_budget)
    return CoVisGraph(frames=frames, edges=edges)


def _apply_pair_budget(
    frames: Sequence[int], edges: list[CoVisEdge], budget: int
) -> list[CoVisEdge]:
    per: dict[int, list[CoVisEdge]] = {f: [] for f in frames}
    for e in edges:
        per[e.i].append(e)
        per[e.j].append(e)
    keep: set[tuple[int, int]] = set()
    for f, es in per.items():
        for e in sorted(es, key=lambda e: e.iom, reverse=True)[:budget]:
            keep.add((e.i, e.j))
    return [e for e in edges if (e.i, e.j) in keep]


# --- adapters from existing pipeline artefacts -----------------------------------------


def covisibility_from_footprints(
    footprints: Mapping[int, object],
    poses: "object",
    *,
    iom_min: float = 0.10,
    view_angle_max_deg: float = 45.0,
    pair_budget: int | None = None,
) -> CoVisGraph:
    """Adapter for the in-build path: `footprints` is the selection `FootprintCache`
    (frame -> GroundFootprint) and `poses` is the selected-candidates DataFrame
    (indexed by frame number, with easting/northing + roll/pitch/yaw_rad columns)."""
    from scene_recon.geometry.extrinsics import CameraPose

    cells_by_frame: dict[int, CellSet] = {}
    positions: dict[int, tuple[float, float]] = {}
    view_dirs: dict[int, np.ndarray] = {}
    for frame, row in poses.iterrows():
        f = int(frame)
        fp = footprints.get(f)
        if fp is None or not getattr(fp, "valid", False) or not fp.cells:
            continue
        cells_by_frame[f] = fp.cells
        positions[f] = (float(row["easting"]), float(row["northing"]))
        # optical axis (forward) in ENU is the 3rd column of R_cam_to_enu.
        view_dirs[f] = CameraPose.from_row(row).R_cam_to_enu()[:, 2]
    return build_covisibility(
        cells_by_frame,
        positions,
        view_dirs=view_dirs,
        iom_min=iom_min,
        view_angle_max_deg=view_angle_max_deg,
        pair_budget=pair_budget,
    )
